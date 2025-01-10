from mmseg.apis import init_model
import torch
import numpy as np
from PIL import Image
from mmseg.apis import MMSegInferencer
from glob import glob
from ktda.datasets import CloudDataset
import numpy as np
from typing import List
import os
from PIL import Image
from ktda import models
from matplotlib import pyplot as plt

def give_color_to_mask(
    mask: Image.Image | np.ndarray, palette: List[int]
) -> Image.Image:
    """
    give color to mask.
    return:
        color_mask: color mask.
    """
    if isinstance(mask, np.ndarray):
        color_mask = Image.fromarray(mask).convert("P")
    else:
        color_mask = mask
    color_mask.putpalette(palette)
    return color_mask

def get_palette() -> List[int]:
    """
    get palette of dataset.
    return:
        palette: list of palette.
    """
    palette = []
    palette_list = CloudDataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette

device = torch.device('cuda:0')
config_path = 'configs/faceseg/swin_base_ours_seg_cls_15_embed_256_cloud_ce.py'
checkpoint_path = 'work_dirs/swin_base_ours_seg_cls_15_embed_256_cloud_ce/best_mIoU_iter_22080.pth'
image_path = 'data/cloud/img_dir/val/wetlands_LC81750732014035LGN00_patch_6144_1536.png'
mask_path = 'data/cloud/ann_dir/val/wetlands_LC81750732014035LGN00_patch_6144_1536.png'
mask = Image.open(mask_path).convert('P')
gt = give_color_to_mask(mask, palette=get_palette())
inferencer = MMSegInferencer(config_path, checkpoint_path, device=device, classes=CloudDataset.METAINFO["classes"],
        palette=CloudDataset.METAINFO["palette"],)

prediction = inferencer.__call__(image_path)["predictions"]
palette = get_palette()
prediction = prediction.astype(np.uint8)
color_mask = give_color_to_mask(prediction, palette=palette)

plt.subplot(1, 3, 1)
plt.imshow(Image.open(image_path))
plt.axis("off")
plt.title("Input")

plt.subplot(1, 3, 2)
plt.imshow(gt)
plt.axis("off")
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(color_mask)
plt.axis("off")
plt.title("Prediction")
plt.savefig(
    f"prediction.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)

