import paddle
import paddle.nn.functional as F
from paddle import nn

import ppsci
from ppsci.loss.base import Loss


# Default values from the paper/common practice for NowcastNet
DEFAULT_DATA_MAX_VALUE = 128.0
DEFAULT_CLIP_MAX = 24.0


def compute_l_wdis(
    y_true, y_pred, data_max_value=DEFAULT_DATA_MAX_VALUE, clip_max=DEFAULT_CLIP_MAX
):
    """
    Computes the weighted L1 distance component L_wdis(y_true, y_pred).
    The weight w(y_true) is calculated as paddle.clip(1.0 + y_true * data_max_value, max=clip_max).
    The loss is paddle.abs(y_true - y_pred) * w_y_true.

    Args:
        y_true (paddle.Tensor): Ground truth tensor. Assumed to be in original scale or
                                 normalized such that y_true * data_max_value gives the intensity.
        y_pred (paddle.Tensor): Predicted tensor, same scale as y_true.
        data_max_value (float, optional): Maximum value for scaling intensity.
                                           Defaults to DEFAULT_DATA_MAX_VALUE.
        clip_max (float, optional): Maximum value for clipping the weight.
                                     Defaults to DEFAULT_CLIP_MAX.

    Returns:
        paddle.Tensor: Element-wise weighted L1 distance.
    """
    weight_y_true = paddle.clip(1.0 + y_true, min=1.0, max=clip_max) # 使用这个
    loss = paddle.abs(y_true - y_pred) * weight_y_true
    return loss


class EvolutionAccumulationLoss(Loss):
    """
    Computes the accumulation loss for the evolution model of NowcastNet.
    J_accum = Σ_t (L_wdis(x_t, (x'_t)_bili) + L_wdis(x_t, x''_t))
    where x_t is ground truth, (x'_t)_bili is advected frame, x''_t is evolved frame.
    The sum is over time steps, and then a mean is taken over batch and spatial dims.
    """

    def __init__(
        self,
        data_max_value=DEFAULT_DATA_MAX_VALUE,
        clip_max=DEFAULT_CLIP_MAX,
        reduction="mean",
        weight=None,
    ):
        super().__init__(reduction, weight)
        self.data_max_value = data_max_value
        self.clip_max = clip_max

    def forward(self, output_dict, label_dict):
        gt_frames = label_dict["gt_frames"]
        advected_frames = output_dict["advected_frames"]
        evolved_frames = output_dict["evolved_frames"]

        if gt_frames.shape != advected_frames.shape or gt_frames.shape != evolved_frames.shape:
            raise ValueError(
                f"Shape mismatch: gt_frames {gt_frames.shape}, "
                f"advected_frames {advected_frames.shape}, "
                f"evolved_frames {evolved_frames.shape}. "
                "Ensure they are (N, T_future, C, H, W)."
            )

        loss_advected = compute_l_wdis(
            gt_frames, advected_frames, self.data_max_value, self.clip_max
        )
        loss_evolved = compute_l_wdis(
            gt_frames, evolved_frames, self.data_max_value, self.clip_max
        )
        
        summed_loss_advected = paddle.sum(loss_advected, axis=1)
        summed_loss_evolved = paddle.sum(loss_evolved, axis=1)
        total_loss_per_sample = summed_loss_advected + summed_loss_evolved
        
        losses = {"loss_accum": total_loss_per_sample}
        return losses


class EvolutionMotionLoss(Loss):
    """
    Computes the motion loss for the evolution model of NowcastNet.
    J_motion = Σ_t (||∇v_t^x ⊙ w(x_t)||_2^2 + ||∇v_t^y ⊙ √w(x_t)||_2^2)
    The sum is over time steps, and then a mean is taken over batch and spatial dims.
    """

    def __init__(
        self,
        data_max_value=DEFAULT_DATA_MAX_VALUE,
        clip_max=DEFAULT_CLIP_MAX,
        reduction="mean",
        weight=None,
    ):
        super().__init__(reduction, weight)
        self.data_max_value = data_max_value
        self.clip_max = clip_max

        sobel_x_kernel = paddle.to_tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype="float32"
        ).unsqueeze(1)
        sobel_y_kernel = paddle.to_tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype="float32"
        ).unsqueeze(1)
        self.register_buffer("sobel_x", sobel_x_kernel, persistable=False)
        self.register_buffer("sobel_y", sobel_y_kernel, persistable=False)

    def _compute_spatial_gradient(self, field_component):
        N, T, H, W = field_component.shape
        reshaped_component = field_component.reshape([N * T, 1, H, W])
        
        grad_x = F.conv2d(reshaped_component, self.sobel_x, padding="same", groups=1)
        grad_y = F.conv2d(reshaped_component, self.sobel_y, padding="same", groups=1)

        grad_x = grad_x.reshape([N, T, H, W])
        grad_y = grad_y.reshape([N, T, H, W])
        return grad_x, grad_y

    def forward(self, output_dict, label_dict):
        motion_vectors = output_dict["motion_vectors"]
        gt_frames = label_dict["gt_frames"]
        
        if gt_frames.shape[2] == 1: # C=1
            gt_frames_for_weight = paddle.squeeze(gt_frames, axis=2)
        else: # C > 1, average over channels for weight
            gt_frames_for_weight = paddle.mean(gt_frames, axis=2, keepdim=False)

        vx = motion_vectors[:, :, 0, :, :]
        vy = motion_vectors[:, :, 1, :, :]

        grad_vx_x, grad_vx_y = self._compute_spatial_gradient(vx)
        grad_vy_x, grad_vy_y = self._compute_spatial_gradient(vy)

        weight_xt = paddle.clip(
            1.0 + gt_frames_for_weight , min=1.0, max=self.clip_max
        )
        
        loss_vx_term = paddle.square(grad_vx_x * weight_xt) + paddle.square(grad_vx_y * weight_xt)
        
        sqrt_weight_xt = paddle.sqrt(weight_xt)
        loss_vy_term = paddle.square(grad_vy_x * sqrt_weight_xt) + paddle.square(grad_vy_y * sqrt_weight_xt)

        total_loss_per_sample = paddle.sum(loss_vx_term + loss_vy_term, axis=1)
        losses = {"loss_motion": total_loss_per_sample}
        return losses


class EvolutionLoss(Loss):
    """
    Combines EvolutionAccumulationLoss and EvolutionMotionLoss.
    J_evolution = J_accum + lambda_motion * J_motion
    """
    def __init__(
        self,
        lambda_motion,
        data_max_value=DEFAULT_DATA_MAX_VALUE,
        clip_max=DEFAULT_CLIP_MAX,
        reduction="mean",
        weight=None,
    ):
        super().__init__(reduction, weight)
        self.lambda_motion = lambda_motion
        
        self.accumulation_loss_fn = EvolutionAccumulationLoss(
            data_max_value=data_max_value,
            clip_max=clip_max,
            reduction=reduction, 
        )
        self.motion_loss_fn = EvolutionMotionLoss(
            data_max_value=data_max_value,
            clip_max=clip_max,
            reduction=reduction, 
        )

    def forward(self, output_dict, label_dict):
        accum_loss_dict = self.accumulation_loss_fn.forward(output_dict, label_dict)
        motion_loss_dict = self.motion_loss_fn.forward(output_dict, label_dict)

        loss_accum = accum_loss_dict["loss_accum"] # (N,C,H,W)
        loss_motion = motion_loss_dict["loss_motion"] # (N,H,W)
        
        squeezed_loss_accum = loss_accum
        if loss_accum.shape[1] == 1 and len(loss_accum.shape) == 4 and len(loss_motion.shape) == 3:
            squeezed_loss_accum = loss_accum.squeeze(axis=1)
        elif loss_accum.shape == loss_motion.shape:
            squeezed_loss_accum = loss_accum
        else:
            raise ValueError(
                f"Shape mismatch for combining accum ({loss_accum.shape}) and motion ({loss_motion.shape}) losses. "
                "Ensure loss_accum is (N,1,H,W) to be squeezed or shapes are identical."
            )
        
        total_evolution_loss = squeezed_loss_accum + self.lambda_motion * loss_motion
        losses = {"loss_evolution": total_evolution_loss}
        return losses


class DiscriminatorLoss(Loss):
    """
    Computes the discriminator loss for NowcastNet's GAN component.
    J_disc = L_ce(D(X_1:T), 1) + L_ce(D(X̃_1:T), 0)
    Assumes D outputs logits.
    """
    def __init__(self, reduction="mean", weight=None):
        super().__init__(reduction, weight)

    def forward(self, output_dict, label_dict):
        disc_real_output = output_dict["disc_real_output"]
        disc_fake_output = output_dict["disc_fake_output"]

        loss_real = F.binary_cross_entropy_with_logits(
            disc_real_output, paddle.ones_like(disc_real_output)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            disc_fake_output, paddle.zeros_like(disc_fake_output)
        )
        loss_disc = loss_real + loss_fake
        losses = {"loss_disc": loss_disc}
        return losses


class GeneratorAdversarialLoss(Loss):
    """
    Computes the adversarial loss for the generator in NowcastNet's GAN component.
    J_adv = L_ce(D(X̃_1:T), 1)
    Assumes D outputs logits.
    """
    def __init__(self, reduction="mean", weight=None):
        super().__init__(reduction, weight)

    def forward(self, output_dict, label_dict):
        disc_fake_output_for_gen = output_dict["disc_fake_output_for_gen_loss"]
        loss_adv = F.binary_cross_entropy_with_logits(
            disc_fake_output_for_gen, paddle.ones_like(disc_fake_output_for_gen)
        )
        losses = {"loss_adv": loss_adv}
        return losses


class PoolRegularizationLoss(Loss):
    """
    Computes the pool regularization loss for NowcastNet.
    J_pool = L_wdis(Q(X_1:T), Q(1/k Σ_i X̃_1:T^(i)))
    where Q is a spatial max pooling layer.
    """
    def __init__(
        self,
        pool_size,
        data_max_value=DEFAULT_DATA_MAX_VALUE,
        clip_max=DEFAULT_CLIP_MAX,
        reduction="mean",
        weight=None,
    ):
        super().__init__(reduction, weight)
        if not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer.")
        self.pool_size = pool_size
        self.data_max_value = data_max_value
        self.clip_max = clip_max

    def _apply_pooling(self, frames_sequence):
        N, T, C, H, W = frames_sequence.shape
        # Reshape for pooling: (N*T*C, 1, H, W)
        reshaped_frames = frames_sequence.reshape([N * T * C, 1, H, W])
        pooled_frames = F.max_pool2d(
            reshaped_frames,
            kernel_size=self.pool_size,
            stride=self.pool_size,
            padding=0,
        )
        # Reshape back: (N, T, C, H_pooled, W_pooled)
        _, _, H_pooled, W_pooled = pooled_frames.shape
        return pooled_frames.reshape([N, T, C, H_pooled, W_pooled])

    def forward(self, output_dict, label_dict):
        real_sequences = label_dict["real_sequences"]
        generated_mean = output_dict["generated_sequences_ensemble_mean"]

        if real_sequences.shape != generated_mean.shape:
             raise ValueError(
                f"Shape mismatch for PoolRegularizationLoss: real_sequences {real_sequences.shape}, "
                f"generated_mean {generated_mean.shape}."
            )

        pooled_real = self._apply_pooling(real_sequences)
        pooled_gen_mean = self._apply_pooling(generated_mean)

        loss_wdis_elements = compute_l_wdis(
            pooled_real, pooled_gen_mean, self.data_max_value, self.clip_max
        )
        loss_pool = paddle.sum(loss_wdis_elements, axis=1) # Sum over time T
        losses = {"loss_pool": loss_pool}
        return losses


class GenerativeNetworkLoss(Loss):
    """
    Combines GeneratorAdversarialLoss and PoolRegularizationLoss for the generator update.
    J_generative = beta_adv * J_adv + gamma_pool * J_pool
    """
    def __init__(
        self,
        beta_adv,
        gamma_pool,
        pool_size, 
        data_max_value=DEFAULT_DATA_MAX_VALUE,
        clip_max=DEFAULT_CLIP_MAX,
        reduction="mean",
        weight=None,
    ):
        super().__init__(reduction, weight)
        self.beta_adv = beta_adv
        self.gamma_pool = gamma_pool

        self.adv_loss_fn = GeneratorAdversarialLoss(reduction=reduction)
        self.pool_loss_fn = PoolRegularizationLoss(
            pool_size=pool_size,
            data_max_value=data_max_value,
            clip_max=clip_max,
            reduction=reduction,
        )

    def forward(self, output_dict, label_dict):
        adv_loss_dict = self.adv_loss_fn.forward(output_dict, label_dict)
        pool_loss_dict = self.pool_loss_fn.forward(output_dict, label_dict)

        loss_adv = adv_loss_dict["loss_adv"] 
        loss_pool = pool_loss_dict["loss_pool"]
        
        # F.binary_cross_entropy_with_logits by default returns a scalar (mean reduced).
        # loss_pool is (N, C, H_pooled, W_pooled) after time summation.
        # So, scalar * scalar + scalar * tensor_N_C_Hp_Wp. This broadcasts fine.
        total_generative_loss = self.beta_adv * loss_adv + self.gamma_pool * loss_pool
        
        losses = {"loss_gen": total_generative_loss}
        return losses


if __name__ == "__main__":
    # --- Testing Evolution Related Losses ---
    N, T_future, C, H, W = 2, 5, 1, 64, 64 
    gt_frames_future = paddle.rand([N, T_future, C, H, W])
    advected_frames = paddle.rand([N, T_future, C, H, W])
    evolved_frames = paddle.rand([N, T_future, C, H, W])
    motion_vectors = paddle.rand([N, T_future, 2, H, W])

    evo_output_dict = {
        "advected_frames": advected_frames,
        "evolved_frames": evolved_frames,
        "motion_vectors": motion_vectors,
    }
    evo_label_dict = {"gt_frames": gt_frames_future}

    accum_loss_fn = EvolutionAccumulationLoss(reduction="mean")
    accum_losses = accum_loss_fn(evo_output_dict, evo_label_dict)
    print(f"EvolutionAccumulationLoss: {accum_losses['loss_accum'].item()}")

    motion_loss_fn = EvolutionMotionLoss(reduction="mean")
    motion_losses = motion_loss_fn(evo_output_dict, evo_label_dict)
    print(f"EvolutionMotionLoss: {motion_losses['loss_motion'].item()}")

    lambda_motion_val = 0.1
    evolution_loss_fn = EvolutionLoss(lambda_motion=lambda_motion_val, reduction="mean")
    evo_losses = evolution_loss_fn(evo_output_dict, evo_label_dict)
    print(f"EvolutionLoss: {evo_losses['loss_evolution'].item()}")
    
    # --- Illustrating reduction behavior ---
    print("\n--- Illustrating reduction behavior (EvolutionLoss) ---")
    raw_accum_loss_tensor = accum_loss_fn.accumulation_loss_fn.forward(evo_output_dict, evo_label_dict)['loss_accum']
    reduced_accum_loss = paddle.mean(raw_accum_loss_tensor)
    print(f"Simulated 'mean' reduction for AccumLoss component: {reduced_accum_loss.item()}")

    raw_motion_loss_tensor = motion_loss_fn.motion_loss_fn.forward(evo_output_dict, evo_label_dict)['loss_motion']
    reduced_motion_loss = paddle.mean(raw_motion_loss_tensor)
    print(f"Simulated 'mean' reduction for MotionLoss component: {reduced_motion_loss.item()}")
    
    # EvolutionLoss.forward combines unreduced components then base class reduces.
    # accum_loss_fn.forward() and motion_loss_fn.forward() are used inside EvolutionLoss.forward()
    raw_evo_components_accum = evolution_loss_fn.accumulation_loss_fn.forward(evo_output_dict, evo_label_dict)['loss_accum']
    raw_evo_components_motion = evolution_loss_fn.motion_loss_fn.forward(evo_output_dict, evo_label_dict)['loss_motion']
    
    squeezed_raw_accum = raw_evo_components_accum
    if raw_evo_components_accum.shape[1] == 1 and len(raw_evo_components_accum.shape) == 4 and len(raw_evo_components_motion.shape) == 3:
         squeezed_raw_accum = raw_evo_components_accum.squeeze(axis=1)
    
    expected_raw_evo_loss = squeezed_raw_accum + lambda_motion_val * raw_evo_components_motion
    expected_reduced_evo_loss = paddle.mean(expected_raw_evo_loss)
    print(f"Expected EvolutionLoss (manual call to sub-forwards and final reduction): {expected_reduced_evo_loss.item()}")


    # --- Testing GAN Related Losses ---
    print("\n--- Testing GAN Related Losses ---")
    N_gan, T_gan, C_gan, H_gan, W_gan = 2, 18, 1, 64, 64 # Example GAN sequence params
    pool_s = 4 # pool_size for PoolRegularizationLoss

    disc_real_logits = paddle.randn([N_gan, 1]) 
    disc_fake_logits = paddle.randn([N_gan, 1])
    disc_fake_logits_for_gen = paddle.randn([N_gan,1])

    real_seq = paddle.rand([N_gan, T_gan, C_gan, H_gan, W_gan])
    gen_seq_mean = paddle.rand([N_gan, T_gan, C_gan, H_gan, W_gan])

    gan_output_dict = {
        "disc_real_output": disc_real_logits,
        "disc_fake_output": disc_fake_logits,
        "disc_fake_output_for_gen_loss": disc_fake_logits_for_gen,
        "generated_sequences_ensemble_mean": gen_seq_mean,
    }
    gan_label_dict = {"real_sequences": real_seq}

    disc_loss_fn = DiscriminatorLoss(reduction="mean")
    disc_losses_val = disc_loss_fn(gan_output_dict, gan_label_dict)
    print(f"DiscriminatorLoss: {disc_losses_val['loss_disc'].item()}")

    gen_adv_loss_fn = GeneratorAdversarialLoss(reduction="mean") 
    adv_losses_val = gen_adv_loss_fn(gan_output_dict, gan_label_dict)
    print(f"GeneratorAdversarialLoss: {adv_losses_val['loss_adv'].item()}")

    pool_reg_loss_fn = PoolRegularizationLoss(pool_size=pool_s, reduction="mean")
    pool_losses_val = pool_reg_loss_fn(gan_output_dict, gan_label_dict)
    print(f"PoolRegularizationLoss: {pool_losses_val['loss_pool'].item()}")

    beta = 0.01
    gamma = 10.0
    gen_net_loss_fn = GenerativeNetworkLoss(
        beta_adv=beta, gamma_pool=gamma, pool_size=pool_s, reduction="mean"
    )
    gen_total_losses_val = gen_net_loss_fn(gan_output_dict, gan_label_dict)
    print(f"GenerativeNetworkLoss: {gen_total_losses_val['loss_gen'].item()}")

    # Manual check for GenerativeNetworkLoss reduction
    print("\n--- Illustrating reduction behavior (GenerativeNetworkLoss) ---")
    # adv_loss_fn.forward() is called in GenerativeNetworkLoss.forward()
    raw_adv_loss_tensor = gen_net_loss_fn.adv_loss_fn.forward(gan_output_dict, gan_label_dict)['loss_adv'] # Scalar from BCE
    # pool_loss_fn.forward() is called in GenerativeNetworkLoss.forward()
    raw_pool_loss_tensor = gen_net_loss_fn.pool_loss_fn.forward(gan_output_dict, gan_label_dict)['loss_pool'] # (N,C,Hp,Wp)
    
    combined_raw_gen_loss = beta * raw_adv_loss_tensor + gamma * raw_pool_loss_tensor
    expected_reduced_total_gen_loss = paddle.mean(combined_raw_gen_loss)
    print(f"Expected GenerativeNetworkLoss (manual call to sub-forwards and final reduction): {expected_reduced_total_gen_loss.item()}")

    print("\nFinished all illustrative examples.")
