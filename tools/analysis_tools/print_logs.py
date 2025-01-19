import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_history(json_file, save_dir):
    # 创建保存图片的目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    # 提取需要的字段名
    parameters = [
        'base_lr', 'lr', 'loss', 'decode.loss_ce', 'decode.loss_dice', 'decode.acc_seg', 
        'aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall'
    ]
    
    # 绘制并保存每个参数的曲线图
    for param in parameters:
        values = []
        for entry in data:
            if param in entry:
                values.append(entry[param])
        
        # 绘图
        if values:
            plt.figure(figsize=(8, 6))
            plt.plot(range(len(values)), values, label=param)
            plt.xlabel('Steps')
            plt.ylabel(param)
            plt.title(f'{param} over training')
            plt.legend()
            
            # 保存图像
            save_path = os.path.join(save_dir, f'{param}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot for {param} to {save_path}")
    
    # 拼接 loss, mIoU, lr 的图像
    loss_values = []
    lr_values = []
    mIoU_values = []
    
    for entry in data:
        if 'loss' in entry:
            loss_values.append(entry['loss'])
        if 'lr' in entry:
            lr_values.append(entry['lr'])
        if 'mIoU' in entry:
            mIoU_values.append(entry['mIoU'])

    # 绘制拼接图
    fig, axes = plt.subplots(3, 1, figsize=(8, 18))  # 3行1列的子图
    axes[0].plot(range(len(loss_values)), loss_values, label='loss', color='blue')
    axes[0].set_title('Loss over training')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(range(len(mIoU_values)), mIoU_values, label='mIoU', color='green')
    axes[1].set_title('mIoU over training')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('mIoU')
    
    axes[2].plot(range(len(lr_values)), lr_values, label='lr', color='red')
    axes[2].set_title('Learning Rate over training')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Learning Rate')
    
    # 自动调整子图间的距离
    plt.tight_layout()

    # 保存拼接图像
    combined_save_path = os.path.join(save_dir, 'loss_mIoU_lr_combined.png')
    plt.savefig(combined_save_path)
    plt.close()
    print(f"Saved combined plot for loss, mIoU, and lr to {combined_save_path}")

parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, default='work_dirs/swinv2_base_ours_seg_cls_15_embed_256_loveDA_train_val_all_ce_dice_feat_fuse_ln_1e-4_32k_data_process_like_sfanet/20250118_220917/vis_data/scalars.json')
parser.add_argument("--output_dir", type=str, default='plot_training_metrics')
args = parser.parse_args()
# 设置JSON文件路径和保存图片的路径
json_file = args.json_file
save_dir = args.output_dir

# 调用函数绘制并保存图像
plot_training_history(json_file, save_dir)

