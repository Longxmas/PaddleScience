# merge_weights.py
import paddle
import os
from collections import OrderedDict


def merge_weights_func(evo_model_path, gan_model_path, save_dir):
    # --- 1. 定义路径 ---
    # 指向第一阶段训练得到的最好/最终模型
    evo_stage_model_path = evo_model_path
    # 指向第二阶段训练得到的最好/最终模型
    gan_stage_model_path = gan_model_path

    # 定义合并后最终模型文件的保存路径
    final_model_save_path = save_dir
    os.makedirs(final_model_save_path, exist_ok=True)
    final_model_save_file = os.path.join(final_model_save_path, "nowcastnet_final.pdparams")

    # --- 2. 加载两个阶段的 state_dict ---
    print("Loading weights from both stages...")
    evo_state_dict = paddle.load(evo_stage_model_path)
    gan_state_dict = paddle.load(gan_stage_model_path)

    # --- 3. 创建一个新的 state_dict 用于存放最终权重 ---
    final_state_dict = OrderedDict()

    # --- 4. 遍历并合并权重 ---
    # 我们以 gan_state_dict 的键为准，因为它是完整的模型结构
    print("Merging weights...")
    for key, param in gan_state_dict.items():
        if key.startswith("evo_net."):
            # 如果是 evo_net 的参数，从第一阶段的权重中获取
            if key in evo_state_dict:
                print(f"  - Taking '{key}' from evolution stage weights.")
                final_state_dict[key] = evo_state_dict[key]
            else:
                # 理论上不应该发生
                print(f"  - WARNING: '{key}' not found in evolution weights, using GAN stage version.")
                final_state_dict[key] = param
        else:
            # 如果是 gen_net (gen_enc, gen_dec, proj) 的参数，直接从第二阶段的权重中获取
            print(f"  - Taking '{key}' from GAN stage weights.")
            final_state_dict[key] = param
            
    # --- 5. 保存最终的模型权重 ---
    paddle.save(final_state_dict, final_model_save_file)
    print(f"\nSuccessfully merged weights and saved to: {final_model_save_file}")