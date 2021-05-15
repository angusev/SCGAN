import torch
import cv2
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import isfile, join

from model import get_generator
from util import arguments
from util.transforms import ToNumpyRGB256
from dataset.InpaintDataset import SCDataset

from metrics import PSNR, SSIM


torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    args = arguments.parse_arguments()
    args.data = Path(args.data)

    net_G = get_generator(args)
    net_G.load_state_dict(torch.load(args.load_G))
    net_G = net_G.cuda().eval()
    torgb = ToNumpyRGB256(-1.0, 1.0)

    imgpath = args.data / "images_256"
    collpath = args.data / "collages"
    respath = args.data / "results"

    collpath.mkdir(exist_ok=True)
    respath.mkdir(exist_ok=True)
    # files = [Path(f).stem for f in listdir(imgpath) if isfile(join(imgpath, f))]

    files = ["100"]

    dataset = SCDataset(args.data, files)

    psnr = 0.0
    ssim = 0.0
    P = PSNR()
    S = SSIM()
    n_elems = len(dataset)
    for i, item in enumerate(tqdm(dataset)):
        image, colormap, sketch, mask = (
            item["image"].unsqueeze(0).cuda(),
            item["colormap"].unsqueeze(0).cuda(),
            item["sketch"].unsqueeze(0).cuda(),
            item["mask"].unsqueeze(0).cuda(),
        )

        generator_input = torch.cat(
            (image * mask, colormap * (1 - mask), sketch * (1 - mask), mask), dim=1
        )
        coarse_image, refined_image = net_G(generator_input)
        completed_image = refined_image * (1 - mask) + image * mask

        metrics_input = torgb(completed_image.squeeze()), torgb(image.squeeze())
        psnr += P(*metrics_input) / n_elems
        ssim += S(*metrics_input) / n_elems

    print("PSNR:", psnr)
    print("ssim:", psnr)