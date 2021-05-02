import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

from util.custom_transforms import UserSimulator


@dataclass()
class ItemsBatch:
    images: torch.Tensor
    colormaps: torch.Tensor
    sketches: torch.Tensor
    masks: torch.Tensor


@dataclass()
class DatasetItem:
    image: Union[torch.Tensor, np.array]
    colormap: Union[torch.Tensor, np.array]
    sketch: Union[torch.Tensor, np.array]
    mask: Union[torch.Tensor, np.array]

    @classmethod
    def collate(cls, items: Sequence["DatasetItem"]) -> ItemsBatch:
        items = list(items)
        return ItemsBatch(
            images=default_collate([item.image for item in items]),
            colormaps=default_collate([item.colormap for item in items]),
            sketches=default_collate([item.sketch for item in items]),
            masks=default_collate([item.mask for item in items]),
        )


class SCDataset(Dataset):
    def __init__(self, data_root, files, sc_only=False):
        self._data_root = Path(data_root)
        self._files = files
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.user_simulator = UserSimulator()
        self.sc_only = sc_only

    def __getitem__(self, index: int) -> DatasetItem:
        pathes = {
            "image": str(
                self._data_root / "images_256" / (self._files[index] + ".jpg")
            ),
            "colormap": str(
                self._data_root / "colormaps_256" / (self._files[index] + ".png")
            ),
            "sketch": str(
                self._data_root / "sketches_masked_256" / (self._files[index] + ".jpg")
            ),
            "mask": self._data_root / "masks_256" / (self._files[index] + ".png"),
        }

        image = (cv2.imread(pathes["image"], -1) / 255 - 0.5) * 2
        colormap = (cv2.imread(pathes["colormap"], -1) / 255 - 0.5) * 2
        sketch = cv2.imread(pathes["sketch"], -1)

        assert image is not None
        assert colormap is not None, f"{pathes['colormap']} does't exist"
        assert sketch is not None
        if self.sc_only:
            mask = np.zeros_like(image[:, :, 0])
        elif pathes["mask"].is_file():
            mask = (cv2.imread(str(pathes["mask"]), -1) / 255)[:, :, None].astype(
                np.float32
            )
        else:
            mask = self.user_simulator(image)

        return dict(
            image=self.transform(image).float(),
            colormap=self.transform(colormap).float(),
            sketch=self.transform(sketch).float(),
            mask=self.transform(mask).float(),
        )

    def __len__(self) -> int:
        return len(self._files)


class SCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 64,
        num_workers: int = 3,
        dry_try: bool = False,
        sc_only: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dry_try = dry_try
        self.sc_only = sc_only

        self.transform = transforms.Compose([transforms.ToTensor()])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)
        # self.num_classes = 10

    def setup(self, stage=None):
        imgpath = self.data_dir / "images_256"
        files = [Path(f).stem for f in listdir(imgpath) if isfile(join(imgpath, f))]

        if self.dry_try:
            files = files[::100]
        dataset = SCDataset(self.data_dir, files, self.sc_only)

        n = len(dataset)
        lengths = [int(n * 0.8), int(n * 0.15), int(n * 0.05)]
        self.train_ds, self.valid_ds, self.test_ds = random_split(dataset, lengths)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
