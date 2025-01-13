from typing import List
from mmseg.apis import init_model
import torch
from torch import nn as nn
from ktda import models
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm
from sklearn.manifold import TSNE

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, filenames:List[str]):
        self.filenames = filenames
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),  # 调整图像尺寸
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize(  # 对图像进行标准化
                mean=[123.675, 116.28, 103.53], 
                std=[58.395, 57.12, 57.375]
            ),
        ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image
        

def plot_tsne(data, labels=None, perplexity=30, learning_rate=200, n_iter=2000, random_state=42):
    """
    使用 t-SNE 对高维数据进行降维并可视化。
    
    参数：
    - data: np.array 或者 pd.DataFrame, 需要降维的数据，形状为 (N, D)
    - labels: 可选，类别标签，若提供则颜色区分
    - perplexity: t-SNE 的困惑度，影响点之间的分布关系
    - learning_rate: t-SNE 的学习率
    - n_iter: t-SNE 的迭代次数
    - random_state: 随机种子，保证结果可复现
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                n_iter=n_iter, random_state=random_state)
    
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='jet', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)

    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.savefig("tsne_visualization.png")

# 使用示例（你需要替换 data 和 labels）
if __name__ == "__main__":
    # 这里你可以用自己的数据替换
    config_path = "work_dirs/swin_base_ours_seg_cls_15_embed_256_cloud_ce_dice_tranform_layer_3/swin_base_ours_seg_cls_15_embed_256_cloud_ce_dice_tranform_layer_3.py"
    checkpoint_path = "work_dirs/swin_base_ours_seg_cls_15_embed_256_cloud_ce_dice_tranform_layer_3/best_mIoU_iter_22080.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model:nn.Module = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    filenames = glob("data/cloud/img_dir/val/*.png")
    datset = CustomDataset(filenames)
    dataloader = torch.utils.data.DataLoader(datset, batch_size=64, shuffle=False)
    if not os.path.exists("hyper_in_feature.npy"):
        print("Generate features")
        hyper_in_feature = None
        for batch in tqdm(dataloader,total=len(dataloader),desc="Generate features"):
            batch = batch.to(device)
            with torch.no_grad():
                hyper_in = model.forward(batch)['hyper_in']
                hyper_in = hyper_in.cpu().numpy()

                if hyper_in_feature is None:
                    hyper_in_feature = hyper_in
                else:
                    hyper_in_feature = np.concatenate([hyper_in_feature,hyper_in],axis=0)
        
        hyper_in_feature = hyper_in_feature.reshape(-1,hyper_in_feature.shape[-1])
        np.save("hyper_in_feature.npy",hyper_in_feature)
    else:
        hyper_in_feature = np.load("hyper_in_feature.npy")
    # print(hyper_in_feature.shape) # hyper_in_feature.shape = (39645, 16)
    plot_tsne(hyper_in_feature)