import os
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
    def __init__(self, data_root, files):
        self._data_root = Path(data_root)
        self._files = files
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.user_simulator = UserSimulator()

    def __getitem__(self, index: int) -> DatasetItem:
        pathes = {
            "image": str(self._data_root / "imgs_256" / self._files[index]),
            "colormap": str(self._data_root / "color_maps_256" / self._files[index]),
            "sketch": str(self._data_root / "sketches" / Path(self._files[index]).with_suffix(".jpg")),
        }

        image = cv2.imread(pathes["image"], -1)
        colormap = cv2.imread(pathes["colormap"], -1)
        sketch = cv2.imread(pathes["sketch"], -1)

        mask = self.user_simulator(image)
        # return DatasetItem(
        #     image=self.transform(image),
        #     colormap=self.transform(colormap),
        #     sketch=self.transform(sketch),
        #     mask=self.transform(mask),
        # )
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
        self, data_dir: str = "./", batch_size: int = 64, num_workers: int = 3
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)
        # self.num_classes = 10

    def setup(self, stage=None):
        files = list((self.data_dir / 'imgs_256').glob("*.png"))
        files = [f.name for f in files]
        dataset = SCDataset(self.data_dir, files)

        n = len(dataset)
        lengths = [int(n * 0.8), int(n * 0.15), int(n * 0.05)]
        self.train_ds, self.valid_ds, self.test_ds = random_split(dataset, lengths)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
