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


torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    args = arguments.parse_arguments()

    net_G = get_generator(args)
    net_G.load_state_dict(torch.load(args.load_G))
    torgb = ToNumpyRGB256(-1., 1.)

    imgpath = args.data_dir / "images_256"
    files = [Path(f).stem for f in listdir(imgpath) if isfile(join(imgpath, f))]

    dataset = SCDataset(args.data_dir, files)
    for i, item in enumerate(tqdm(dataset)):
        item = item.cuda()
        image, colormap, sketch, mask = (
            item["image"].unsqueeze(0),
            item["colormap"].unsqueeze(0),
            item["sketch"].unsqueeze(0),
            item["mask"].unsqueeze(0),
        )

        generator_input = torch.cat(
            (image * mask, colormap * (1 - mask), sketch * (1 - mask), mask), dim=1
        )
        coarse_image, refined_image = net_G(generator_input)
        completed_image = (
            (refined_image * (1 - mask) + image * mask)
        )

        masked_image = image * mask
        masked_sketch = sketch * (1 - mask)
        masked_colormap = colormap * (1 - mask)

        visualization = [
            torgb(masked_image.squeeze().cpu().numpy()),
            torgb(masked_sketch.squeeze().cpu().numpy()),
            torgb(masked_colormap.squeeze().cpu().numpy()),
            torgb(completed_image.squeeze().cpu().numpy()),
            torgb(image.squeeze().cpu().numpy()),
        ]
        image_fromarray = Image.fromarray(np.hstack(visualization)[:, :, [2, 1, 0]])
        image_fromarray.save(args.data_dir / 'collages' / files[i])
        cv2.write(str(args.data_dir / 'results' / files[i]), visualization[2])
