import os
from glob import glob
import argparse
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--raw_path",
        type=str,
        default="data/uavid/uavid_v1.5_official_release_image/uavid_v1.5_official_release_image",
    )
    parse.add_argument("--save_dir", type=str, default="data/uavid")
    args = parse.parse_args()
    return args


def get_images(raw_path):
    # data/uavid/uavid_v1.5_official_release_image/uavid_v1.5_official_release_image/uavid_train/seq6/Images/000100.png
    images = glob(os.path.join(raw_path, "*", "*", "Images", "*"))
    return images

def convert_to_single_channel(image_path,save_path):
    # 颜色到标签的映射字典
    color_to_label = {
        (0, 0, 0): 0,       # Background clutter
        (128, 0, 0): 1,     # Building
        (128, 64, 128): 2,  # Road
        (0, 128, 0): 3,     # Tree
        (128, 128, 0): 4,   # Low vegetation
        (64, 0, 128): 5,    # Moving car
        (192, 0, 192): 6,   # Static car
        (64, 64, 0): 7      # Human
    }

    # 打开图像并转换为NumPy数组
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # 创建一个与输入图像同大小的空白单通道图像
    single_channel_image = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

    # 遍历每个像素并转换为对应的标签值
    for color, label in color_to_label.items():
        mask = np.all(image_array == np.array(color), axis=-1)
        single_channel_image[mask] = label
    
    # 保存单通道图像
    Image.fromarray(single_channel_image).save(save_path)


def save_file(image_path, save_dir, is_image=True):
    split = image_path.split(os.path.sep)[-4].split("_")[-1]
    seq = image_path.split(os.path.sep)[-3]
    basename = os.path.basename(image_path)
    new_basename = f"{seq}_{basename}"
    if is_image:
        save_path = os.path.join(save_dir, "img_dir", split, new_basename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(image_path, save_path)
    else:
        save_path = os.path.join(save_dir, "ann_dir", split, new_basename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        convert_to_single_channel(image_path,save_path)



def main():
    args = get_args()
    images = get_images(args.raw_path)
    for image in tqdm(images):
        save_file(image, args.save_dir)
        save_file(image.replace("Images","Labels"), args.save_dir, is_image=False)


if __name__ == "__main__":
    main()
