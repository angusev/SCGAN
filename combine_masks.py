import argparse
import os
import glob

from pathlib import Path
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from joblib import Parallel, delayed

import cv2
import numpy as np


def get_format(f):
    return f.split(".")[-1]


INDEX_CLASS_MAPPING = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "eye_g",
    4: "l_eye",
    5: "r_eye",
    6: "l_brow",
    7: "r_brow",
    8: "l_ear",
    9: "r_ear",
    10: "mouth",
    11: "u_lip",
    12: "l_lip",
    13: "hair",
    14: "hat",
    15: "ear_r",
    16: "neck_l",
    17: "neck",
    18: "cloth",
}
CLASS_INDEX_MAPPING = {v: k for k, v in INDEX_CLASS_MAPPING.items()}
N_CLASSES = len(CLASS_INDEX_MAPPING)

MASK_SIZE = (512, 512)


def process_one_file(f, masks_path, dest_path):
    wo_format = f.split(".")[0]
    n = int(wo_format)
    fileprefix = wo_format.rjust(5, "0")

    combination = np.zeros(MASK_SIZE)
    folder = masks_path / str(n // 2000)
    mask_files = glob.glob(str(folder / f"{fileprefix}_*.png"))
    parts = [Path(mask_file).name[6:-4] for mask_file in mask_files]
    for idx, part in INDEX_CLASS_MAPPING.items():
        if part in parts:
            mask = (
                cv2.imread(
                    str(folder / f"{fileprefix}_{part}.png"), cv2.IMREAD_GRAYSCALE
                )
                / 255
            )
            combination[mask != 0] = idx

    new_file_path = dest_path / f"{n}.png"
    new_file_path.parents[0].mkdir(exist_ok=True)
    cv2.imwrite(str(new_file_path), combination)


def combine_masks(args):
    imgs_path = Path(args.imgs)
    src_path = Path(args.masks)
    dest_path = Path(args.dest)
    dest_path.mkdir(exist_ok=True)

    # r=root, d=directories, f = files
    files = [f for f in listdir(imgs_path) if isfile(join(imgs_path, f))]

    Parallel(n_jobs=4, verbose=10)(
        delayed(process_one_file)(f, src_path, dest_path) for f in files
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)

    args = parser.parse_args()
    combine_masks(args)
