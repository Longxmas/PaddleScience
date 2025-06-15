import os
import time
import paddle
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig # 导入 DictConfig 用于类型提示

from ppsci.arch import NowcastNet
from ppsci.data.dataset import RadarDataset

# =======================================================
# 1. 评估指标计算函数 (移植自 forcast.py)
# =======================================================

def cal_tp(pred, target, th):
    return paddle.where(
        (pred >= th) & (target >= th), 
        paddle.ones_like(target), 
        paddle.zeros_like(target)
    ).sum(axis=(-1, -2))

def cal_fp(pred, target, th):
    return paddle.where(
        (pred >= th) & (target < th),
        paddle.ones_like(target),
        paddle.zeros_like(target)
    ).sum(axis=(-1, -2))

def cal_fn(pred, target, th):
    return paddle.where(
        (pred < th) & (target >= th),
        paddle.ones_like(target),
        paddle.zeros_like(target)
    ).sum(axis=(-1, -2))

def cal_csi(pred, target, threshold=16.0):
    """Calculates the Critical Success Index (CSI)."""
    tp = cal_tp(pred=pred, target=target, th=threshold)
    fp = cal_fp(pred=pred, target=target, th=threshold)
    fn = cal_fn(pred=pred, target=target, th=threshold)
    
    tp_fp_fn = tp + fp + fn
    csi = tp / paddle.where(tp_fp_fn > 1e-5, tp_fp_fn, paddle.full_like(tp_fp_fn, 1e-5))
    return csi.mean(axis=0)

# =======================================================
# 2. 可视化函数 (移植并改进自 plt_img)
# =======================================================

def get_alpha_mask(image_data, threshold=2.0):
    """Creates a transparency mask."""
    alpha = np.zeros_like(image_data, dtype=float)
    alpha[image_data >= threshold] = 1.0
    return alpha

# 在文件的第2部分，替换原来的 plot_results 函数

def plot_results(
    last_input_frame,
    future_gt,
    future_pred_evo,  # Evo-Net only prediction
    future_pred_gen,  # Final prediction from Gen-Net
    time_indices,
    interval_minutes=10,
    save_path="visualization.png",
    vmin=0,
    vmax=40,
    cmap="viridis"
):
    """
    Visualizes and saves a comparison of GT, Evo-Net, and Gen-Net results.
    Layout is always 3 rows (GT, Evo-Only, Final-Gen) x 4 columns for generation eval.
    For evolution-only eval, it will be 2 rows.
    """
    if len(time_indices) != 3:
        raise ValueError("time_indices must contain exactly 3 indices.")

    is_full_eval = future_pred_gen is not None
    num_rows = 3 if is_full_eval else 2
    figsize_height = 18 if is_full_eval else 12

    fig, axs = plt.subplots(num_rows, 4, figsize=(24, figsize_height), squeeze=False)
    fig.suptitle("Prediction vs. Ground Truth Comparison", fontsize=20)

    # --- Helper function for plotting a row ---
    def plot_row(row_idx, row_label, data, input_frame):
        # Plot input frame
        ax_input = axs[row_idx, 0]
        ax_input.imshow(input_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_input.set_title("Input at t=0")
        # Use Y-axis label to identify the row content
        ax_input.set_ylabel(row_label, fontsize=16, weight='bold', rotation=0, labelpad=80, verticalalignment='center')
        ax_input.set_yticklabels([])
        ax_input.set_xticklabels([])
        ax_input.tick_params(axis='both', which='both', length=0)


        # Plot future frames
        for col, t_idx in enumerate(time_indices, 1):
            ax = axs[row_idx, col]
            if t_idx >= data.shape[0]:
                ax.set_title(f"(Index {t_idx} OOB)")
                ax.axis('off')
                continue

            img = data[t_idx]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            rgba_img = plt.get_cmap(cmap)(plt.Normalize(vmin=vmin, vmax=vmax)(img))
            rgba_img[..., -1] = get_alpha_mask(img)

            ax.imshow(rgba_img)
            time_min = (t_idx + 1) * interval_minutes
            ax.set_title(f"t+{time_min} min")
            ax.axis('off')

    # --- Plot rows ---
    plot_row(0, "Ground Truth", future_gt, last_input_frame)
    plot_row(1, "Evo-Net Only", future_pred_evo, last_input_frame)
    if is_full_eval:
        plot_row(2, "NowcastNet", future_pred_gen, last_input_frame)

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")


# =======================================================
# 3. EvolutionEvaluator 类
# =======================================================

class EvolutionEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        
        self.data_params = cfg.get("TRAIN") # Use train paths for now, can be specified
        self.eval_params = cfg.get("EVAL")
        self.model_params = cfg.get("MODEL").get(cfg.CASE_TYPE)
        
        self.t_in = self.model_params.input_length
        self.t_out = self.model_params.total_length - self.model_params.input_length
        self.csi_threshold = self.eval_params.csi_threshold
        self.time_interval = self.data_params.get("interval_minutes", 10) # Add this to yaml if needed
        
        self.output_dir = "eval_output_evolution"
        self.vis_save_path = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.vis_save_path, exist_ok=True)

    def _load_model(self):
        """Loads the model and sets the weights."""
        print("Loading model and weights...")
        self.model = NowcastNet(**self.model_params)
        
        if not self.eval_params.pretrained_model_path:
            raise ValueError("EVAL.pretrained_model_path must be set in the config!")
        
        state_dict = paddle.load(self.eval_params.pretrained_model_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded weights from {self.eval_params.pretrained_model_path}")

    def print_csi_metrics(self, csi_scores, info_prefix="CSI"):
        """Logs the CSI scores for key timesteps."""
        metrics_info = [info_prefix]
        key_timesteps = self.eval_params.key_info_timestep
        for timestep_min in key_timesteps:
            idx = timestep_min // self.time_interval - 1
            if 0 <= idx < len(csi_scores):
                metrics_info.append(f"T+{timestep_min}min: {csi_scores[idx]:.4f}")
        print(" ".join(metrics_info))

    def run(self):
        """Main evaluation loop."""
        self._load_model()
        print("================= Starting Evolution Net Evaluation =================")
        
        eval_dataset = RadarDataset(
            input_keys=("radar_frames",),
            label_keys=(),
            dataset_path=self.data_params.VALID_DATA_PATH,
            image_width=self.model_params.image_width,
            image_height=self.model_params.image_height,
            total_length=self.model_params.total_length,
        )
        eval_dataloader = paddle.io.DataLoader(
            eval_dataset,
            batch_size=self.eval_params.batch_size,
            shuffle=False,
            num_workers=self.eval_params.num_workers
        )
        
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        total_csi_scores = []
        
        for batch_id, data in enumerate(eval_dataloader):
            if batch_id >= self.eval_params.get("num_eval_samples", 5): # Limit samples to evaluate
                print(f"Reached evaluation limit of {self.eval_params.num_eval_samples} samples.")
                break

            t1 = time.time()
            input_tensor = data[0]['radar_frames']
            
            with paddle.no_grad():
                pred_tensor = self.model.forward_tensor_evo_only(input_tensor)
            
            gt_tensor = input_tensor[:, self.t_in:, ..., 0]
            csi_scores_batch = cal_csi(pred=pred_tensor, target=gt_tensor, threshold=self.csi_threshold)
            total_csi_scores.append(csi_scores_batch)
            self.print_csi_metrics(csi_scores_batch.numpy(), info_prefix=f"Batch {batch_id} CSI")
            
            if self.eval_params.visualize:
                # 从 batch 中取出第一个样本
                gt_sample = gt_tensor[0].numpy()
                pred_sample = pred_tensor[0].numpy()
                
                # 【修改点】: 获取 t 时刻的输入帧 (即 past_frames 的最后一帧)
                last_input_frame_sample = input_tensor[0, self.t_in - 1, ..., 0].numpy()
                
                time_indices_to_plot = [t // self.time_interval - 1 for t in self.eval_params.key_info_timestep]
                
                # 【修改点】: 调用新的 plot_results 函数
                plot_results(
                    last_input_frame=last_input_frame_sample,
                    future_gt=gt_sample,
                    future_pred_evo=pred_sample, # Evo-only prediction
                    future_pred_gen=None,        # No Gen-Net prediction in this mode
                    time_indices=time_indices_to_plot,
                    save_path=os.path.join(self.vis_save_path, f"evolution_batch_{batch_id}.png"),
                    interval_minutes=self.time_interval
                )


            step_cost = (time.time() - t1) * 1000
            print(f"Batch {batch_id}, Inference Cost: {step_cost:.2f} ms")

        if total_csi_scores:
            avg_csi_scores = paddle.stack(total_csi_scores).mean(axis=0)
            self.print_csi_metrics(avg_csi_scores.numpy(), info_prefix="Average CSI over all evaluated batches")
        
        print("================== Finished Evolution Net Evaluation ==================")


# =======================================================
# 4. GenerationEvaluator 类 (新添加)
# =======================================================

class GenerationEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        
        self.data_params = cfg.get("TRAIN")
        self.eval_params = cfg.get("EVAL")
        self.model_params = cfg.get("MODEL").get(cfg.CASE_TYPE)
        
        self.t_in = self.model_params.input_length
        self.t_out = self.model_params.total_length - self.model_params.input_length
        self.csi_threshold = self.eval_params.csi_threshold
        self.time_interval = self.data_params.get("interval_minutes", 10)
        
        # 使用不同的输出目录以区分
        self.output_dir = "eval_output_generation"
        self.vis_save_path = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.vis_save_path, exist_ok=True)

    def _load_model(self):
        """Loads the model and sets the weights."""
        print("Loading model and weights for full network evaluation...")
        self.model = NowcastNet(**self.model_params)
        
        if not self.eval_params.pretrained_model_path:
            raise ValueError("EVAL.pretrained_model_path must be set in the config!")
        
        state_dict = paddle.load(self.eval_params.pretrained_model_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded weights from {self.eval_params.pretrained_model_path}")

    def print_csi_metrics(self, csi_scores, info_prefix="CSI"):
        """Logs the CSI scores for key timesteps."""
        metrics_info = [info_prefix]
        key_timesteps = self.eval_params.key_info_timestep
        for timestep_min in key_timesteps:
            idx = timestep_min // self.time_interval - 1
            if 0 <= idx < len(csi_scores):
                metrics_info.append(f"T+{timestep_min}min: {csi_scores[idx]:.4f}")
        print(" ".join(metrics_info))

    def run(self):
        """Main evaluation loop for the full generation network."""
        self._load_model()
        print("================= Starting Full Network (Generation) Evaluation =================")
        
        eval_dataset = RadarDataset(
            input_keys=("radar_frames",),
            label_keys=(),
            dataset_path=self.data_params.VALID_DATA_PATH,
            image_width=self.model_params.image_width,
            image_height=self.model_params.image_height,
            total_length=self.model_params.total_length,
        )
        eval_dataloader = paddle.io.DataLoader(
            eval_dataset,
            batch_size=self.eval_params.batch_size,
            shuffle=False,
            num_workers=self.eval_params.num_workers
        )
        
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        total_csi_evo = []
        total_csi_gen = []
        
        for batch_id, data in enumerate(eval_dataloader):
            if batch_id >= self.eval_params.get("num_eval_samples", 5):
                print(f"Reached evaluation limit of {self.eval_params.num_eval_samples} samples.")
                break

            t1 = time.time()
            input_tensor = data[0]['radar_frames']
            gt_tensor = input_tensor[:, self.t_in:, ..., 0]
            
            with paddle.no_grad():
                # --- Step 1: Get Evo-Net only prediction ---
                pred_evo = self.model.forward_tensor_evo_only(input_tensor)
                
                # --- Step 2: Get Final (Gen-Net) prediction ---
                input_dict = {"radar_frames": input_tensor}
                output_dict = self.model(input_dict)
                pred_gen = output_dict[self.model.output_keys[0]]

            # --- Calculate CSI for both predictions ---
            # CSI for Evo-Net only
            csi_evo_batch = cal_csi(pred=pred_evo, target=gt_tensor, threshold=self.csi_threshold)
            total_csi_evo.append(csi_evo_batch)
            self.print_csi_metrics(csi_evo_batch.numpy(), info_prefix=f"Batch {batch_id} Evo-Only CSI")
            
            
            if pred_gen.shape[-1] == 1:
                pred_gen_squeezed = paddle.squeeze(pred_gen, axis=-1)
            else:
                pred_gen_squeezed = pred_gen
                
            # CSI for Final Gen-Net result
            csi_gen_batch = cal_csi(pred=pred_gen_squeezed, target=gt_tensor, threshold=self.csi_threshold)
            total_csi_gen.append(csi_gen_batch)
            self.print_csi_metrics(csi_gen_batch.numpy(), info_prefix=f"Batch {batch_id} Final-Gen CSI")
            
            # --- Visualize all three results ---
            if self.eval_params.visualize:
                gt_sample = gt_tensor[0].numpy()
                pred_evo_sample = pred_evo[0].numpy()
                pred_gen_sample = pred_gen[0].numpy()
                last_input_frame_sample = input_tensor[0, self.t_in - 1, ..., 0].numpy()
                
                time_indices_to_plot = [t // self.time_interval - 1 for t in self.eval_params.key_info_timestep]
                
                plot_results(
                    last_input_frame=last_input_frame_sample,
                    future_gt=gt_sample,
                    future_pred_evo=pred_evo_sample,  # Pass Evo-only prediction
                    future_pred_gen=pred_gen_sample,  # Pass Final prediction
                    time_indices=time_indices_to_plot,
                    save_path=os.path.join(self.vis_save_path, f"generation_comparison_batch_{batch_id}.png"),
                    interval_minutes=self.time_interval
                )

            step_cost = (time.time() - t1) * 1000
            print(f"Batch {batch_id}, Full Inference Cost: {step_cost:.2f} ms\n")

        # --- Print Average CSI for both ---
        if total_csi_evo:
            avg_csi_evo = paddle.stack(total_csi_evo).mean(axis=0)
            self.print_csi_metrics(avg_csi_evo.numpy(), info_prefix="Average Evo-Only CSI")
        if total_csi_gen:
            avg_csi_gen = paddle.stack(total_csi_gen).mean(axis=0)
            self.print_csi_metrics(avg_csi_gen.numpy(), info_prefix="Average Final-Gen CSI")
        
        print("================== Finished Full Network (Generation) Evaluation ==================")


# =======================================================
# 5. Main function to run the evaluation independently
# =======================================================

import argparse # 导入 argparse

def main():
    """
    Main entry point for running the evaluation script.
    Allows choosing between 'evolution' and 'generation' evaluation modes.
    """
    # --- 添加命令行参数解析 ---
    parser = argparse.ArgumentParser(description="NowcastNet Evaluation Script")
    parser.add_argument(
        '--mode',
        type=str,
        default='generation',
        choices=['generation', 'evolution'],
        help="Evaluation mode: 'generation' for the full network, 'evolution' for the evolution net only."
    )
    args = parser.parse_args()
    
    # --- Configuration ---
    CONFIG_PATH = "./conf/nowcastnet.yaml"
    
    try:
        cfg = OmegaConf.load(CONFIG_PATH)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        return

    # --- Override specific evaluation settings if needed ---
    # cfg.EVAL.pretrained_model_path = "path/to/your/model.pdparams"
    
    print("--- Current Evaluation Configuration ---")
    print(f"Mode: {args.mode.upper()}")
    print(OmegaConf.to_yaml(cfg.EVAL))
    print("----------------------------------------")
    
    # --- Instantiate and run the chosen evaluator ---
    if args.mode == 'generation':
        evaluator = GenerationEvaluator(cfg)
    elif args.mode == 'evolution':
        evaluator = EvolutionEvaluator(cfg)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
        
    evaluator.run()

if __name__ == "__main__":
    main()