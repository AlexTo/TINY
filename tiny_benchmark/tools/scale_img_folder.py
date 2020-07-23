import argparse
import numpy as np
import os

from PIL import Image
from ISR.models import RDN
from pathlib import Path


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="psnr-small", required=False)
    parser.add_argument("--scale", type=int, default=2, required=False)

    return parser.parse_args()


def super_res(src_dir, dest_dir, model):
    rdn = RDN(weights=model)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(src_dir):
        if os.path.exists(f"{dest_dir}/{f}"):
            continue
        img = Image.open(f"{src_dir}/{f}")
        img = np.array(img)
        sr_img = rdn.predict(img, by_patch_of_size=50)
        sr_img = Image.fromarray(sr_img)
        sr_img.save(f"{dest_dir}/{f}")


if __name__ == '__main__':
    args = init_args()
    super_res(args.src_dir, args.dest_dir, args.model)
