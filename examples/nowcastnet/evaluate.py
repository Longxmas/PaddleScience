import os
import os.path as osp

import numpy as np
import paddle
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# 新增导入，用于手动实现邻域计算
from scipy.ndimage import uniform_filter
from pysteps.utils.spectral import rapsd

from scipy import fft
import hydra
import ppsci
from ppsci.utils import logger

# -------------------------- 辅助函数 (核心修改在这里) --------------------------

def neighborhood_csi(pred, obs, thr, neighborhood_size):
    """
    手动计算邻域CSI。

    Args:
        pred (np.array): 预测场 (H, W)
        obs (np.array): 观测场 (H, W)
        thr (float): 降水阈值
        neighborhood_size (int): 邻域窗口的边长 (e.g., 5 for 5x5)

    Returns:
        float: 邻域CSI得分
    """
    # 1. 二值化
    pred_bin = (pred >= thr).astype(float)
    obs_bin = (obs >= thr).astype(float)

    # 2. 邻域平滑 (计算邻域分数)
    # uniform_filter等价于用一个值为1/N^2的卷积核进行卷积
    pred_scores = uniform_filter(pred_bin, size=neighborhood_size, mode='constant', cval=0)
    obs_scores = uniform_filter(obs_bin, size=neighborhood_size, mode='constant', cval=0)

    # 3. 计算二元列联表 (hits, misses, false_alarms)
    # 只要邻域分数大于0，就认为该点有事件
    hits = np.sum((pred_scores > 0) & (obs_scores > 0))
    misses = np.sum((pred_scores == 0) & (obs_scores > 0))
    false_alarms = np.sum((pred_scores > 0) & (obs_scores == 0))

    # 4. 计算CSI
    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan  # 或者返回0，取决于定义
    csi = hits / denominator
    return csi



def calculate_csi_and_psd(pred_frames, true_frames, thresholds, psd_frame_indices, input_length):
    """
    计算单个样本的邻域CSI和PSD。
    """
    csi_scores = {thr: [] for thr in thresholds}
    psd_scores = {
        "pred": {idx: None for idx in psd_frame_indices},
        "true": {idx: None for idx in psd_frame_indices}
    }
    
    neighborhood_size = 5

    num_lead_times = pred_frames.shape[0]
    for i in range(num_lead_times):
        pred_frame = pred_frames[i]
        true_frame_value = true_frames[i, :, :, 0]
        for thr in thresholds:
            csi = neighborhood_csi(pred_frame, true_frame_value, thr, neighborhood_size)
            csi_scores[thr].append(csi)

    psd_local_indices = [idx - input_length for idx in psd_frame_indices]
    freqs = None
    for local_idx, global_idx in zip(psd_local_indices, psd_frame_indices):
        if 0 <= local_idx < pred_frames.shape[0]:
            pred_frame = pred_frames[local_idx]
            true_frame_value = true_frames[local_idx, :, :, 0]

            # --- 核心修改：使用rapsd函数 ---
            # d=1.0因为我们后面会根据分辨率手动调整波长
            psd_pred, freqs_current = rapsd(pred_frame, fft_method=fft, return_freq=True, d=1.0)
            psd_true, _ = rapsd(true_frame_value, fft_method=fft, return_freq=True, d=1.0)
            # ------------------------------------
            
            psd_scores["pred"][global_idx] = psd_pred
            psd_scores["true"][global_idx] = psd_true
            if freqs is None:
                freqs = freqs_current
    
    if freqs is None:
        # 如果所有psd_frame_indices都无效，则模拟一个freqs
        # --- 核心修改：使用rapsd函数 ---
        _, freqs = rapsd(pred_frames[0], fft_method=fft, return_freq=True, d=1.0)
        # ------------------------------------

    resolution_km_per_pixel = 2.0
    # freqs的单位是 1/pixel，所以 1/freqs 是波长（单位：像素）
    # 当频率为0时，1/freqs会导致除零错误，需要处理
    non_zero_freqs = freqs != 0
    wavelengths = np.full_like(freqs, np.inf) # 频率为0对应无限波长
    wavelengths[non_zero_freqs] = (1 / freqs[non_zero_freqs]) * resolution_km_per_pixel

    return csi_scores, psd_scores, wavelengths

def plot_case_study(pred_frames, true_frames, case_name, output_dir):
    """
    为单个样本绘制定性对比图 (2x3 网格)，样式匹配论文（白底）。

    Args:
        pred_frames (np.array): 模型的18个或更多预测帧 (T_out, H, W)。
        true_frames (np.array): 真实的18个或更多未来帧 (T_out, H, W, 2)。
        case_name (str): 案例名称，用于文件名 (e.g., "case_0")。
        output_dir (str): 输出目录。
    """
    local_indices_to_plot = [5, 11, 17]
    time_titles = ["T + 1 h", "T + 2 h", "T + 3 h"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    
    cmap = 'viridis'
    vmin, vmax = 0, 40

    for i, (local_idx, title) in enumerate(zip(local_indices_to_plot, time_titles)):
        # --- 绘制观测图 (第一行) ---
        ax_obs = axes[0, i]
        true_frame_val = true_frames[local_idx, :, :, 0]
        # 使用imshow绘制降水场
        im = ax_obs.imshow(true_frame_val, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        ax_obs.set_title(title, fontsize=14)
        
        # --- 核心修改：设置白色背景和隐藏边框 ---
        ax_obs.set_facecolor('white') # 设置背景为白色
        ax_obs.set_xticks([])
        ax_obs.set_yticks([])
        for spine in ax_obs.spines.values(): # 隐藏外边框
            spine.set_visible(False)
        # ----------------------------------------
        
        # --- 绘制预测图 (第二行) ---
        ax_pred = axes[1, i]
        pred_frame_val = pred_frames[local_idx]
        ax_pred.imshow(pred_frame_val, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

        # --- 核心修改：设置白色背景和隐藏边框 ---
        ax_pred.set_facecolor('white') # 设置背景为白色
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        for spine in ax_pred.spines.values(): # 隐藏外边框
            spine.set_visible(False)
        # ----------------------------------------

    axes[0, 0].set_ylabel("Observations", fontsize=14)
    axes[1, 0].set_ylabel("NowcastNet", fontsize=14)
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Precipitation (mm h⁻¹)")

    # 移除tight_layout，使用subplots_adjust进行更精细的控制
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
    
    save_path = osp.join(output_dir, f"{case_name}.png")
    plt.savefig(save_path, dpi=300, facecolor='white') # 保存时也指定背景色
    plt.close(fig)
    logger.info(f"Case study plot saved to {save_path}")

def plot_csi(csi_results, output_dir, num_lead_times):
    """绘制CSI曲线图，样式匹配论文。"""
    lead_times_steps = np.arange(1, num_lead_times + 1)
    
    # --- 创建两个子图，对应 >=16 和 >=32 的阈值 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # 阈值列表，假设按顺序
    thresholds = sorted(csi_results.keys())
    
    for i, thr in enumerate(thresholds):
        ax = axes[i]
        scores = csi_results[thr]
        ax.plot(lead_times_steps, scores, marker='o', linestyle='-', markersize=4)
        
        ax.set_title(f"Precipitation (mm h⁻¹) ≥ {int(thr)}")
        ax.set_xlabel("Prediction interval (10 min)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # --- 核心修改：设置坐标轴 ---
        ax.set_ylim(0, 0.8)
        ax.set_xticks(np.arange(0, num_lead_times + 1, 3))
        ax.set_xlim(0, num_lead_times)

    axes[0].set_ylabel("CSI neighbourhood")
    
    plt.tight_layout()
    save_path = osp.join(output_dir, "csi_evaluation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"CSI plot saved to {save_path}")

def plot_psd(psd_results, wavelengths, output_dir, psd_frame_indices, input_length):
    """绘制PSD曲线图，样式匹配论文。"""
    
    # --- 创建两个子图，对应 T+2h 和 T+3h ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    indices_to_plot = psd_frame_indices[1:]
    
    for i, global_idx in enumerate(indices_to_plot):
        if global_idx not in psd_results["true"] or psd_results["true"][global_idx] is None:
            continue
            
        ax = axes[i]
        lead_time_hr = (global_idx - (input_length - 1)) * 10 / 60
        
        # --- 核心修改：将PSD转换为dB单位 ---
        # 10 * log10(PSD)，并处理可能为0的PSD值
        psd_true_db = 10 * np.log10(psd_results["true"][global_idx] + 1e-10)
        psd_pred_db = 10 * np.log10(psd_results["pred"][global_idx] + 1e-10)
        
        ax.plot(wavelengths, psd_true_db, linestyle='--', color='black', label='Ground truth')
        ax.plot(wavelengths, psd_pred_db, label='NowcastNet')

        ax.set_title(f"T + {lead_time_hr:.0f} h")
        ax.set_xlabel("Wavelength (km)")
        
        # --- 核心修改：设置坐标轴 ---
        ax.set_xscale('log')
        ax.invert_xaxis() # 反转X轴
        
        # 设置X轴刻度为2的幂次方
        ticks = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter()) # 使用常规数字格式
        ax.tick_params(axis='x', which='minor', bottom=False) # 关闭次要刻度
        
        # 设置Y轴范围和网格
        ax.set_ylim(-20, 70)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel("PSD")
    axes[1].legend() # 只在第二个图上显示图例
    
    plt.tight_layout()
    save_path = osp.join(output_dir, "psd_evaluation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"PSD plot saved to {save_path}")


# -------------------------- 最终的 evaluate 函数 (方案A) --------------------------
def evaluate(cfg: DictConfig):
    """
    对NowcastNet模型进行定量评估，计算并绘制CSI和PSD指标。
    """
    # 1. 初始化和配置加载
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    if cfg.CASE_TYPE == "large":
        dataset_path = cfg.LARGE_DATASET_PATH
        model_cfg = cfg.MODEL.large
        output_dir = osp.join(cfg.output_dir, "large_eval")
    elif cfg.CASE_TYPE == "normal":
        dataset_path = cfg.NORMAL_DATASET_PATH
        model_cfg = cfg.MODEL.normal
        output_dir = osp.join(cfg.output_dir, "normal_eval")
    else:
        raise ValueError(
            f"cfg.CASE_TYPE should be in ['normal', 'large'], but got '{cfg.mode}'"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    model = ppsci.arch.NowcastNet(**model_cfg)

    # 2. 数据加载器设置
    input_keys = ("radar_frames",)
    
    # 核心修改：确保数据加载与模型配置匹配
    # 假设模型配置中明确定义了输入和输出长度
    # 如果没有，需要根据日志硬编码
    try:
        input_length = model_cfg.input_length
        # 从日志看，模型输出了20帧
        output_length = 20 # model_cfg.output_length
    except AttributeError:
        # 如果model_cfg中没有这些属性，根据情况设定
        input_length = 9
        output_length = 20
        logger.warning(
            f"model_cfg lacks input/output_length, using default values: input={input_length}, output={output_length}"
        )

    # total_length现在由模型配置决定
    total_length = input_length + output_length # 应该等于 29
    logger.info(f"Model config: Input={input_length}, Output={output_length}, Total={total_length}")
    
    dataset_param = {
        "input_keys": input_keys,
        "label_keys": (),
        "image_width": model_cfg.image_width,
        "image_height": model_cfg.image_height,
        "total_length": total_length, # 使用修正后的total_length
        "dataset_path": dataset_path,
        "data_type": paddle.get_default_dtype(),
    }
    test_data_loader = paddle.io.DataLoader(
        ppsci.data.dataset.RadarDataset(**dataset_param),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.CPU_WORKER,
        drop_last=True,
    )

    # 3. Solver 和评估指标初始化
    solver = ppsci.solver.Solver(
        model,
        output_dir=output_dir,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    
    thresholds = [16.0, 32.0]
    # T的时刻是第 input_length 帧结束时
    # T+1h = 60min = 6帧。全局索引 = (input_length - 1) + 6 = 8 + 6 = 14
    # T+2h = 120min = 12帧。全局索引 = 8 + 12 = 20
    # T+3h = 180min = 18帧。全局索引 = 8 + 18 = 26
    psd_frame_indices = [input_length + 5, input_length + 11, input_length + 17] # [14, 20, 26]

    all_csi_scores = {thr: [] for thr in thresholds}
    all_psd_scores_pred = {idx: [] for idx in psd_frame_indices}
    all_psd_scores_true = {idx: [] for idx in psd_frame_indices}
    
    num_samples = len(test_data_loader)
    logger.info(f"Start evaluation on {num_samples} samples...")
    
    # 4. 评估循环
    for batch_id, data in enumerate(test_data_loader):
        if (batch_id + 1) % 10 == 0:
            logger.info(f"Evaluating sample {batch_id + 1}/{num_samples}")

        frames_tensor = data[0][input_keys[0]]
        
        output_dict = solver.predict(input_dict={input_keys[0]: frames_tensor})
        pred_frames = output_dict['output'].squeeze().numpy() 
        
        true_frames_all = frames_tensor.squeeze().numpy()
        true_frames_future = true_frames_all[input_length:]
        
        # 现在 pred_frames 和 true_frames_future 的长度应该都等于 output_length (20)
        if pred_frames.shape[0] != true_frames_future.shape[0]:
            logger.warning(
                f"FATAL: Mismatch after correction. Pred: {pred_frames.shape[0]}, True: {true_frames_future.shape[0]}. slicing."
            )
            
        plot_case_study(
            pred_frames, # 传入完整的预测帧
            true_frames_future, # 传入完整的真实未来帧
            case_name=f"case_study_{batch_id}",
            output_dir=output_dir,
        )
        
        # 论文只评估到T+3h (18帧), 所以我们只评估前18帧
        eval_frames = 18
        pred_to_eval = pred_frames[:eval_frames]
        true_to_eval = true_frames_future[:eval_frames]
        
        # 计算CSI和PSD
        csi_scores, psd_scores, wavelengths = calculate_csi_and_psd(
            pred_to_eval, true_to_eval, thresholds, psd_frame_indices, input_length
        )

        # 累积结果
        for thr in thresholds:
            all_csi_scores[thr].append(csi_scores[thr])
        for idx in psd_frame_indices:
            if psd_scores["pred"][idx] is not None:
                all_psd_scores_pred[idx].append(psd_scores["pred"][idx])
                all_psd_scores_true[idx].append(psd_scores["true"][idx])
    
    if not all_csi_scores[thresholds[0]]:
        logger.error("No samples were evaluated successfully. Exiting.")
        return

    logger.info("Evaluation finished. Aggregating and plotting results...")
    
    mean_csi = {thr: np.mean(np.array(scores), axis=0) for thr, scores in all_csi_scores.items()}
    mean_psd = {
        "pred": {idx: np.mean(np.array(all_psd_scores_pred[idx]), axis=0) for idx in psd_frame_indices if all_psd_scores_pred[idx]},
        "true": {idx: np.mean(np.array(all_psd_scores_true[idx]), axis=0) for idx in psd_frame_indices if all_psd_scores_true[idx]}
    }
    
    for thr, scores in mean_csi.items():
        logger.info(f"Mean CSI @ {thr}mm/h over 18 lead times: {np.mean(scores):.4f}")
    
    plot_csi(mean_csi, output_dir, num_lead_times=18)
    plot_psd(mean_psd, wavelengths, output_dir, psd_frame_indices, input_length)
    
    logger.info(f"All evaluation plots have been saved to {output_dir}")

@hydra.main(version_base=None, config_path="./conf", config_name="nowcastnet.yaml")
def main(cfg: DictConfig):
    evaluate(cfg)

if __name__ == "__main__":
    main()