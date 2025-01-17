from glob import glob
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default='work_dirs/format_results')
    return parser.parse_args()

def main():
    args = get_args()
    filenames = glob(os.path.join(args.input_dir, '*.png'))
    for filename in tqdm(filenames,desc="post processing"):
        im = np.array(Image.open(filename))
        im -= 1
        Image.fromarray(im).save(filename)


if __name__ == "__main__":
    main()