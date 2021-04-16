import argparse
import os
from pathlib import Path
from shutil import copyfile
from joblib import Parallel, delayed

import cv2


def get_format(f):
    return f.split(".")[-1]


def compress_and_save(f, dataset_path, compressed_path, size):
    form = get_format(f)
    old_file_path = dataset_path / f
    new_file_path = compressed_path / f
    new_file_path = new_file_path.with_suffix('.png')
    new_file_path.parents[0].mkdir(exist_ok=True)
    
    im = cv2.imread(str(old_file_path))
    im = cv2.resize(im, (size, size), cv2.INTER_AREA)
    cv2.imwrite(str(new_file_path), im)


def compress_dataset(dataset_path, compressed_path, size):
    dataset_path = Path(dataset_path)
    compressed_path = Path(compressed_path)
    compressed_path.mkdir(exist_ok=True)

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dataset_path):
        for f_i in f:
            file_path = os.path.join(r, f_i)
            files.append(os.path.relpath(file_path, dataset_path))

    Parallel(n_jobs=4, verbose=10)(
        delayed(compress_and_save)(f, dataset_path, compressed_path, size)
        for f in files
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the dataset", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--new_path",
        type=str,
        default=None,
        help="path for saving a compressed dataset",
    )

    args = parser.parse_args()
    if args.new_path is None:
        args.new_path = f'{args.path.rstrip("/")}_{args.size}/'
    compress_dataset(args.path, args.new_path, args.size)
