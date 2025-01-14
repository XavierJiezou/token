from mmseg.apis import MMSegInferencer, init_model
from glob import glob
from rich.progress import track
from ktda import models
import numpy as np
import argparse
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from typing import List
from mmseg.datasets import LoveDADataset
from natsort import natsorted
from skimage.io import imsave

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default="work_dirs/swinv2_base_ours_seg_cls_15_embed_256_loveDA_ce_dice/best_mIoU_iter_20470.pth")
    parse.add_argument("--save_path", type=str, default="data/loveDA/test_dir")
    parse.add_argument("--config_path", type=str, default="work_dirs/swinv2_base_ours_seg_cls_15_embed_256_loveDA_ce_dice/swinv2_base_ours_seg_cls_15_embed_256_loveDA_ce_dice.py")
    parse.add_argument("--device", type=str, default="cuda")
    parse.add_argument("--dataset_path", type=str, default="data/loveDA")
    parse.add_argument("--batch_size", type=int, default=16)
    args = parse.parse_args()
    return args

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb

def save_mask(mask, image_path):
    mask = Image.fromarray(label2rgb(mask))
    mask.save(image_path)


def get_images(args):
    # data/loveDA/img_dir/test
    image_paths = glob(f"{args.dataset_path}/img_dir/test/*.png")
    images = natsorted(image_paths)
    return images

def get_infereser(args):
    inferencer = MMSegInferencer(args.config_path, args.weight_path, device=args.device)
    return inferencer
def get_palette() -> List[int]:
    """
    get palette of dataset.
    return:
        palette: list of palette.
    """
    palette = []
    palette_list = LoveDADataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette


def give_color_to_mask(
    mask: Image.Image | np.ndarray, palette: List[int]
) -> Image.Image:
    """
    give color to mask.
    return:
        color_mask: color mask.
    """
    color_mask = Image.fromarray(mask).convert("P")
    color_mask.putpalette(palette)
    return color_mask
def main():
    args = get_args()
    inferencer = get_infereser(args)
    images = get_images(args)
    # dataLoader = get_dataloader(args)
    os.makedirs(args.save_path, exist_ok=True)
    total_length = len(images)
    palette = get_palette()
    for idx in range(0, total_length, args.batch_size):
        began = idx
        ended = idx + args.batch_size
        if ended > total_length:
            ended = total_length
        file_names = images[began:ended]
        results = inferencer(file_names, batch_size=args.batch_size)["predictions"]
        for result, image_path in zip(results, file_names):
            prediction = result
            prediction = prediction.astype(np.uint8)
            # mask_RGB = label2rgb(prediction)
            # image = give_color_to_mask(prediction, palette)
            base_name = os.path.basename(image_path)
            save_path = os.path.join(args.save_path, base_name)
            # image.save(save_path)
            imsave(save_path, prediction)
            


    # inferencer.__call__(images, batch_size=args.batch_size, out_dir=args.save_path, with_labels=False)
    # palette = get_palette()
    # with torch.no_grad():
    #     for batch in track(dataLoader, description="inferencing...", total=len(dataLoader)):
    #         images, image_paths = batch
    #         images = images.to(args.device)
    #         masks = inferencer(images)["seg_output"]
    #         masks = masks.argmax(dim=1)
 
    #         masks = masks.cpu().numpy().astype(np.uint8)
    #         for mask, image_path in zip(masks, image_paths):
    #             base_name = os.path.basename(image_path)
    #             save_path = os.path.join(args.save_path, base_name)
    #             # save_mask(mask, save_path)
    #             color_mask = give_color_to_mask(mask, palette)
    #             color_mask.save(save_path)

if __name__ == "__main__":
    main()


