import os
import os.path as osp
import json # 确保 json 被导入

import hydra
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger
# 假设你的 loss.py 和 nowcastnet_train.py 在同一目录下
from loss import (
    EvolutionLoss,
    DiscriminatorLoss,
    GeneratorAdversarialLoss,
    PoolRegularizationLoss,
)


class DiscriminatorBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, down_scale=False):
        super().__init__()
        self.bn = nn.BatchNorm2D(in_channels)
        stride = 2 if down_scale else 1
        
        # one_conv 分支
        self.one_conv = nn.utils.spectral_norm(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        )
        
        # double_conv 分支
        self.double_conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.ReLU(),
            nn.utils.spectral_norm(
                nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            )
        )
        
    def forward(self, x):
        bn_x = self.bn(x)
        x1 = self.one_conv(bn_x)
        x2 = self.double_conv(x) # 注意：这里输入的是原始的 x
        return x1 + x2

class TemporalDiscriminator(nn.Layer):
    """
    Temporal Discriminator 的 PaddlePaddle 实现.
    架构严格对齐 MindSpore 版本，使用 Conv2D。
    它将整个时间序列作为输入通道。
    """
    def __init__(self, input_time_length, image_height, image_width, in_channels=1):
        super().__init__()
        # input_time_length 应该等于 model.total_length
        # in_channels 应该等于 model.image_ch (但你的模型里只用了1个通道)
        self.total_input_channels = input_time_length * in_channels
        
        # MindSpore 源码中 hardcode 了这几个值
        hidden1, hidden2, hidden3 = 64, 84, 40

        self.conv1 = nn.Conv2D(self.total_input_channels, hidden1, kernel_size=9, stride=2, padding=4)
        self.conv2 = nn.Conv2D(self.total_input_channels, hidden2, kernel_size=9, stride=2, padding=4)
        self.conv3 = nn.Conv2D(self.total_input_channels, hidden3, kernel_size=9, stride=2, padding=4)
        
        block_in_channels = hidden1 + hidden2 + hidden3
        self.block1 = DiscriminatorBlock(block_in_channels, 128, down_scale=True)
        self.block2 = DiscriminatorBlock(128, 256, down_scale=True)
        self.block3 = DiscriminatorBlock(256, 512, down_scale=True)
        self.block4 = DiscriminatorBlock(512, 512, down_scale=False)
        
        self.bn_final = nn.BatchNorm2D(512)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.final_conv = nn.utils.spectral_norm(
            nn.Conv2D(512, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Input x: (B, T, H, W) or (B, T, C, H, W)
        # 判别器期望的输入是整个序列，在通道维度上被展平
        if len(x.shape) == 5: # B, T, C, H, W
            # Reshape to (B, T*C, H, W)
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        elif len(x.shape) == 4: # B, T, H, W (assumes C=1)
            # 形状正确，无需改变
            pass
        else:
            raise ValueError(f"Unsupported input shape for TemporalDiscriminator: {x.shape}")
        
        # 初始并行卷积
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # 拼接
        out = paddle.concat([x1, x2, x3], axis=1)
        
        # 序贯块
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        # 最终层
        out = self.leaky_relu(self.bn_final(out))
        out = self.final_conv(out)
        return out

# ==============================================================================
# 2. 演化网络训练函数 (新)
# ==============================================================================
def train_evolution(cfg: DictConfig):
    """
    Handles the training for the Evolution Network (Phase 1).
    Corresponds to MindSpore's EvolutionTrainer.
    """
    logger.info("Starting Evolution Network Training Phase...")

    # --- Setup: Model, Optimizer, Loss ---
    if cfg.CASE_TYPE == "large":
        model_cfg = cfg.MODEL.large
    else:
        model_cfg = cfg.MODEL.normal
    
    # In this phase, we only need the evolution network part of NowcastNet.
    # We instantiate the full model for simplicity and weight compatibility.
    model = ppsci.arch.NowcastNet(**model_cfg)

    optimizer_evo = paddle.optimizer.Adam(
        parameters=model.evo_net.parameters(),
        learning_rate=cfg.TRAIN.learning_rate_evo,
        weight_decay=cfg.TRAIN.evol_weight_decay,
        # Mindspore version uses weight_decay, add if needed:
        # weight_decay=cfg.TRAIN.get('weight_decay_evo', 0.0) 
    )

    evolution_loss_fn = EvolutionLoss(
        lambda_motion=cfg.TRAIN.lambda_motion,
        data_max_value=cfg.MODEL.data_max_value,
        clip_max=cfg.MODEL.clip_max,
        reduction='mean'
    )
    
    logger.info("Evolution model, optimizer, and loss function created.")

    # --- Checkpoint Loading ---
    start_epoch = 1
    if cfg.TRAIN.checkpoint_path and osp.exists(osp.join(cfg.TRAIN.checkpoint_path, "model.pdparams")):
        model_load_path = osp.join(cfg.TRAIN.checkpoint_path, "model.pdparams")
        opt_load_path = osp.join(cfg.TRAIN.checkpoint_path, "opt_evo.pdopt")
        metadata_load_path = osp.join(cfg.TRAIN.checkpoint_path, "metadata.json")
        
        model.set_state_dict(paddle.load(model_load_path))
        optimizer_evo.set_state_dict(paddle.load(opt_load_path))
        if osp.exists(metadata_load_path):
            with open(metadata_load_path, 'r') as f:
                metadata = json.load(f)
            start_epoch = metadata.get('epoch', 1) + 1
            logger.info(f"Loaded metadata, resuming from epoch {start_epoch}")
        logger.info(f"Loaded Evolution checkpoint from {cfg.TRAIN.checkpoint_path}")
    else:
        logger.info("No Evolution checkpoint found, starting from scratch.")
    
    train_dataset = ppsci.data.dataset.RadarDataset(
        input_keys=model.input_keys, # ("radar_frames",)
        label_keys=(),
        image_width=model_cfg.image_width,
        image_height=model_cfg.image_height,
        total_length=model_cfg.total_length,
        dataset_path=cfg.TRAIN.TRAIN_DATA_PATH,
        data_type=paddle.get_default_dtype(),
    )
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.evol_batch_size,
        shuffle=False,
        num_workers=cfg.TRAIN.num_workers,
        drop_last=True,
    )
    # Validation dataloader can be added here similarly if needed for evo-net

    # --- Training Loop ---
    num_batches = len(train_dataloader)
    print(f"num_batches = {num_batches}")
    for epoch_id in range(start_epoch, cfg.TRAIN.epochs_evo + 1):
        model.evo_net.train() # Only evo_net needs to be in train mode
        epoch_total_loss = 0.0
        for batch_id, batch_data in enumerate(train_dataloader):
            input_frames_full = batch_data[0][model.input_keys[0]]
            actual_frames = input_frames_full[..., 0] # Shape: B, TotalLen, H, W
            
            # Prepare inputs
            past_frames_evo_input = actual_frames[:, :model.input_length]
            future_frames_gt = actual_frames[:, model.input_length:model.total_length]
            
            optimizer_evo.clear_grad()
            
            # --- Forward Pass and Manual Reconstruction (with gradient detachment) ---
            # This part directly mimics the logic in Mindspore's `EvolutionLoss.construct`
            predicted_intensity_evo, predicted_motion_evo = model.evo_net(past_frames_evo_input)
            
            # Reshape for easier indexing
            current_batch_size = past_frames_evo_input.shape[0]
            predicted_motion_evo_r = predicted_motion_evo.reshape((
                current_batch_size, model.pred_length, 2, model_cfg.image_height, model_cfg.image_width
            ))
            predicted_intensity_evo_r = predicted_intensity_evo.reshape((
                current_batch_size, model.pred_length, 1, model_cfg.image_height, model_cfg.image_width
            ))
            
            advected_frames_list = []
            evolved_frames_list = []
            
            # Start evolution from the last input frame
            current_frame_for_evolution = past_frames_evo_input[:, -1:, :, :] # B, 1, H, W
            
            # Grid for warping
            grid = model.grid.tile((current_batch_size, 1, 1, 1))

            for t in range(model.pred_length):
                current_motion = predicted_motion_evo_r[:, t] # B, 2, H, W
                current_intensity_residual = predicted_intensity_evo_r[:, t] # B, 1, H, W
                
                # Advected frame (pure motion)
                # NOTE: The original NowcastNet warp function is complex. Using a simplified
                # grid_sample version. You might need to align this with ppsci.arch.nowcast.warp
                # For simplicity, we use the `model.forward`'s warp logic as a template.
                vgrid = grid + current_motion
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(model_cfg.image_width - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(model_cfg.image_height - 1, 1) - 1.0
                vgrid = vgrid.transpose((0, 2, 3, 1))
                
                advected_frame_t = F.grid_sample(
                    current_frame_for_evolution, vgrid, mode='bilinear', padding_mode='border', align_corners=True
                )
                
                # Evolved frame (motion + intensity change)
                evolved_frame_t = advected_frame_t + current_intensity_residual
                
                advected_frames_list.append(advected_frame_t)
                evolved_frames_list.append(evolved_frame_t)
                
                # CRITICAL: Detach the gradient as in Mindspore's `ops.stop_gradient`
                # This prevents backpropagation through the entire time sequence.
                current_frame_for_evolution = evolved_frame_t.detach()

            # --- Loss Calculation and Backward Pass ---
            # The EvolutionLoss class will handle combining all components.
            advected_frames_for_loss = paddle.concat(advected_frames_list, axis=1).unsqueeze(2) # B,T,1,H,W
            evolved_frames_for_loss = paddle.concat(evolved_frames_list, axis=1).unsqueeze(2) # B,T,1,H,W

            output_dict = {
                'advected_frames': advected_frames_for_loss,
                'evolved_frames': evolved_frames_for_loss,
                'motion_vectors': predicted_motion_evo_r
            }
            label_dict = {
                'gt_frames': future_frames_gt.unsqueeze(2) # B, T, 1, H, W
            }
            
            loss_dict = evolution_loss_fn(output_dict, label_dict)
            unreduced_loss = loss_dict['loss_evolution']
            total_loss = paddle.mean(unreduced_loss)
            
            total_loss.backward()
            optimizer_evo.step()
            
            epoch_total_loss += total_loss.item()

            if batch_id % cfg.TRAIN.log_freq == 0:
                logger.info(
                    f"Evo-Train | Epoch {epoch_id}/{cfg.TRAIN.epochs_evo}, Batch {batch_id+1}/{num_batches}, "
                    f"Loss: {total_loss.item():.4f}, LR: {optimizer_evo.get_lr():.2e}"
                )
        avg_epoch_loss = epoch_total_loss / num_batches
        logger.info(
            f"Evo-Train | Epoch {epoch_id}/{cfg.TRAIN.epochs_evo} Finished | "
            f"Average Loss: {avg_epoch_loss:.4f}, "
            f"Current LR: {optimizer_evo.get_lr():.2e}"
        )

        # --- End of Epoch Saving ---
        if epoch_id % cfg.TRAIN.save_freq == 0 or epoch_id == cfg.TRAIN.epochs_evo:
            save_dir = osp.join(cfg.output_dir, "checkpoints_evo", f"epoch_{epoch_id}")
            os.makedirs(save_dir, exist_ok=True)
            paddle.save(model.state_dict(), osp.join(save_dir, "model.pdparams"))
            paddle.save(optimizer_evo.state_dict(), osp.join(save_dir, "opt_evo.pdopt"))
            with open(osp.join(save_dir, "metadata.json"), 'w') as f:
                json.dump({'epoch': epoch_id}, f)
            logger.info(f"Saved evolution checkpoint to {save_dir}")

# ==============================================================================
# 3. 生成对抗网络训练函数 (新)
# ==============================================================================
def train_generation(cfg: DictConfig):
    """
    Handles the training for the GAN (Generator + Discriminator) (Phase 2).
    Corresponds to MindSpore's GenerationTrainer.
    """
    logger.info("Starting Generation Network (GAN) Training Phase...")
    
    # --- Setup: Models, Optimizers, Loss Fns ---
    if cfg.CASE_TYPE == "large":
        model_cfg = cfg.MODEL.large
    else:
        model_cfg = cfg.MODEL.normal
    
    model_gen = ppsci.arch.NowcastNet(**model_cfg) # The full generator model
    model_disc = TemporalDiscriminator( # <--- 使用新的判别器
        input_time_length=model_gen.total_length,
        image_height=model_cfg.image_height,
        image_width=model_cfg.image_width,
        in_channels=1
    )

    # Load pre-trained evolution network weights
    # The path should point to the checkpoint from the evolution training phase.
    evo_ckpt_path = cfg.TRAIN.get('evolution_checkpoint_path', None)
    if evo_ckpt_path and osp.exists(evo_ckpt_path):
        state_dict = paddle.load(evo_ckpt_path)
        # It's better to load only evo_net weights if possible to avoid conflicts
        # but loading the full dict is fine if keys match.
        model_gen.set_state_dict(state_dict)
        logger.info(f"Loaded pre-trained evolution weights from {evo_ckpt_path}")
    else:
        logger.warning("No pre-trained evolution checkpoint provided. GAN training may fail.")

    # Freeze the evolution network parameters
    for param in model_gen.evo_net.parameters():
        param.stop_gradient = True
    logger.info("Froze parameters of the Evolution Network (evo_net).")

    # Optimizers for Generator (gen_enc, gen_dec, proj) and Discriminator
    gen_params = (
        list(model_gen.gen_enc.parameters()) + 
        list(model_gen.gen_dec.parameters()) + 
        list(model_gen.proj.parameters())
    )
    optimizer_gen = paddle.optimizer.Adam(
        parameters=gen_params,
        learning_rate=cfg.TRAIN.learning_rate_gen,
        beta1=cfg.TRAIN.get('beta1', 0.9),
        beta2=cfg.TRAIN.get('beta2', 0.999),
    )
    optimizer_disc = paddle.optimizer.Adam(
        parameters=model_disc.parameters(),
        learning_rate=cfg.TRAIN.learning_rate_disc,
        beta1=cfg.TRAIN.get('beta1', 0.9),
        beta2=cfg.TRAIN.get('beta2', 0.999),
    )

    # Loss Functions
    discriminator_loss_fn = DiscriminatorLoss()
    generator_adv_loss_fn = GeneratorAdversarialLoss()
    pool_reg_loss_fn = PoolRegularizationLoss(
        pool_size=cfg.TRAIN.pool_size,
        data_max_value=cfg.MODEL.data_max_value,
        clip_max=cfg.MODEL.clip_max
    )
    
    logger.info("GAN models, optimizers, and loss functions created.")

    # --- Checkpoint Loading (for GAN phase) ---
    start_epoch = 1
    # You might want a different checkpoint path for the GAN phase
    gan_ckpt_path = cfg.TRAIN.get('gan_checkpoint_path', None)
    if gan_ckpt_path and osp.exists(osp.join(gan_ckpt_path, "model_gen.pdparams")):
        model_gen.set_state_dict(paddle.load(osp.join(gan_ckpt_path, "model_gen.pdparams")))
        model_disc.set_state_dict(paddle.load(osp.join(gan_ckpt_path, "model_disc.pdparams")))
        optimizer_gen.set_state_dict(paddle.load(osp.join(gan_ckpt_path, "opt_gen.pdopt")))
        optimizer_disc.set_state_dict(paddle.load(osp.join(gan_ckpt_path, "opt_disc.pdopt")))
        with open(osp.join(gan_ckpt_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        start_epoch = metadata.get('epoch', 1) + 1
        logger.info(f"Loaded GAN checkpoint, resuming from epoch {start_epoch}")
    else:
        logger.info("No GAN checkpoint found, starting GAN training from scratch.")
        
    # --- DataLoader ---
    # The dataset for generation phase might be different if evo_result is pre-computed.
    # Here we assume it's computed on-the-fly with no_grad.
    train_dataset = ppsci.data.dataset.RadarDataset(
        input_keys=model_gen.input_keys,
        label_keys=(),
        image_width=model_cfg.image_width,
        image_height=model_cfg.image_height,
        total_length=model_cfg.total_length,
        dataset_path=cfg.TRAIN.TRAIN_DATA_PATH,
        data_type=paddle.get_default_dtype(),
    )
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.gan_batch_size,
        shuffle=False,
        num_workers=cfg.TRAIN.num_workers,
        drop_last=True,
    )

    # --- Training Loop ---
    num_batches = len(train_dataloader)
    for epoch_id in range(start_epoch, cfg.TRAIN.epochs_gan + 1):
        model_gen.train()
        model_disc.train()
        # Ensure evo_net is in eval mode as it's frozen and has batchnorm layers
        model_gen.evo_net.eval()
        
        epoch_total_loss = 0
        for batch_id, batch_data in enumerate(train_dataloader):
            input_frames_full = batch_data[0][model_gen.input_keys[0]]
            actual_frames = input_frames_full[..., 0] # B, TotalLen, H, W
            
            past_frames = actual_frames[:, :model_gen.input_length]
            future_frames_gt = actual_frames[:, model_gen.input_length:]

            # --- On-the-fly calculation of evo_result with no_grad ---
            # This is done because evo_net is frozen.
            with paddle.no_grad():
                evo_result_unnormalized = model_gen.forward_tensor_evo_only(input_frames_full) # B, T_pred, H, W
                # The model's forward_tensor already normalizes, but let's be explicit
                # Assuming forward_tensor_evo_only returns unnormalized pixel values
                evo_result_normalized = evo_result_unnormalized / cfg.MODEL.data_max_value
            
            # --- Train Generator ---
            optimizer_gen.clear_grad()
            
            # Ensemble generation for Pool Regularization and Adversarial Loss
            # Based on Mindspore: k for pool, 1 for adv. Total k+1 noises.
            ensemble_k = cfg.TRAIN.ensemble_k
            noise = paddle.randn(shape=[
                past_frames.shape[0], model_gen.ngf, 
                model_cfg.image_height // 32, model_cfg.image_width // 32,
                ensemble_k + 1 # +1 for the adversarial loss sample
            ])
            
            # Forward pass for Adversarial Loss (using noise[..., 0])
            gen_output_adv = model_gen.forward_tensor_gen_only(
                input_frames_full, evo_result_normalized, noise[..., 0]
            ).squeeze(-1) # B, T_pred, H, W
            
            # Create full sequence for discriminator
            full_sequence_fake_adv = paddle.concat([past_frames, gen_output_adv], axis=1)
            disc_fake_output_for_gen = model_disc(full_sequence_fake_adv)
            
            loss_adv_dict = generator_adv_loss_fn({'disc_fake_output_for_gen_loss': disc_fake_output_for_gen}, {})
            loss_adv = loss_adv_dict['loss_adv']
            
            # Forward pass for Pool Regularization Loss (using noise[..., 1:])
            ensemble_outputs = []
            for i in range(ensemble_k):
                gen_output_pool_i = model_gen.forward_tensor_gen_only(
                    input_frames_full, evo_result_normalized, noise[..., i+1]
                ).squeeze(-1)
                ensemble_outputs.append(gen_output_pool_i)
            
            ensemble_mean = paddle.mean(paddle.stack(ensemble_outputs, axis=0), axis=0) # B,T_pred,H,W
            
            # Pool loss needs full sequences
            full_sequence_real = actual_frames # B, TotalLen, H, W
            full_sequence_fake_mean = paddle.concat([past_frames, ensemble_mean], axis=1)
            
            output_dict_pool = {'generated_sequences_ensemble_mean': full_sequence_fake_mean.unsqueeze(2)} # Add C dim
            label_dict_pool = {'real_sequences': full_sequence_real.unsqueeze(2)} # Add C dim

            loss_pool_dict = pool_reg_loss_fn(output_dict_pool, label_dict_pool)
            unreduced_loss_pool = loss_pool_dict['loss_pool']
            loss_pool = paddle.mean(unreduced_loss_pool)

            total_loss_gen = cfg.TRAIN.beta_adv * loss_adv + cfg.TRAIN.gamma_pool * loss_pool
            total_loss_gen.backward()
            optimizer_gen.step()
            
            # --- Train Discriminator ---
            optimizer_disc.clear_grad()
            
            # Re-use the generated sample for adversarial loss, but detached.
            disc_real_output = model_disc(full_sequence_real)
            disc_fake_output = model_disc(full_sequence_fake_adv.detach())
            
            loss_disc_dict = discriminator_loss_fn(
                {'disc_real_output': disc_real_output, 'disc_fake_output': disc_fake_output}, {}
            )
            unreduced_loss_disc = loss_disc_dict['loss_disc']
            total_loss_disc = paddle.mean(unreduced_loss_disc)
            
            total_loss_disc.backward()
            optimizer_disc.step()

            # --- Logging ---
            if batch_id % cfg.TRAIN.log_freq == 0:
                logger.info(
                    f"GAN-Train | Epoch {epoch_id}/{cfg.TRAIN.epochs_gan}, Batch {batch_id+1}/{num_batches} | "
                    f"G_Loss: {total_loss_gen.item():.4f} (Adv: {loss_adv.item():.4f}, Pool: {loss_pool.item():.4f}) | "
                    f"D_Loss: {total_loss_disc.item():.4f}"
                )

        # --- End of Epoch Saving ---
        if epoch_id % cfg.TRAIN.save_freq == 0 or epoch_id == cfg.TRAIN.epochs_gan:
            save_dir = osp.join(cfg.output_dir, "checkpoints_gan", f"epoch_{epoch_id}")
            os.makedirs(save_dir, exist_ok=True)
            paddle.save(model_gen.state_dict(), osp.join(save_dir, "model_gen.pdparams"))
            paddle.save(model_disc.state_dict(), osp.join(save_dir, "model_disc.pdparams"))
            paddle.save(optimizer_gen.state_dict(), osp.join(save_dir, "opt_gen.pdopt"))
            paddle.save(optimizer_disc.state_dict(), osp.join(save_dir, "opt_disc.pdopt"))
            with open(osp.join(save_dir, "metadata.json"), 'w') as f:
                json.dump({'epoch': epoch_id}, f)
            logger.info(f"Saved GAN checkpoint to {save_dir}")

# ==============================================================================
# 4. 主入口函数 (修改后)
# ==============================================================================
@hydra.main(version_base=None, config_path="./conf", config_name="train.yaml")
def main(cfg: DictConfig):
    # Set up logger and random seed
    ppsci.utils.misc.set_random_seed(cfg.seed)
    log_file_path = osp.join(cfg.output_dir, f"train_{cfg.TRAIN.module_name}.log")
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.init_logger("ppsci", log_file_path, "info")

    if cfg.TRAIN.module_name == "evolution":
        train_evolution(cfg)
    elif cfg.TRAIN.module_name == "generation":
        train_generation(cfg)
    else:
        raise ValueError(
            f"cfg.TRAIN.module_name should be 'evolution' or 'generation', but got '{cfg.TRAIN.module_name}'"
        )

if __name__ == "__main__":
    main()