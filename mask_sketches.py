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


def process_one_file(f, bin_path, sketches_path, dest_path):
    wo_format = f.split('.')[0]
    n = int(wo_format)
    fileprefix = wo_format.rjust(5, '0')

    sketch = 255 - cv2.imread(str(sketches_path / f), cv2.IMREAD_GRAYSCALE)
    binary_mask = cv2.imread(str((bin_path / f).with_suffix('.png')), -1)
    binary_mask[binary_mask != 0] = 1
    binary_mask = cv2.resize(binary_mask, sketch.shape[:2], cv2.INTER_AREA)

    cv2.imwrite(str(dest_path / f), sketch  * binary_mask)


def combine_masks(args):
    sketches_path = Path(args.sketches)
    bin_path = Path(args.binmasks)
    dest_path = Path(args.dest)
    dest_path.mkdir(exist_ok=True)

    # r=root, d=directories, f = files
    files = [f for f in listdir(sketches_path) if isfile(join(sketches_path, f))]

    Parallel(n_jobs=4, verbose=10)(
        delayed(process_one_file)(f, bin_path, sketches_path, dest_path)
        for f in files
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketches", type=str, required=True)
    parser.add_argument("--binmasks", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)

    args = parser.parse_args()
    combine_masks(args)
