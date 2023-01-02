import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from detectron2.data import DatasetMapper
from .transforms import get_transforms

from util import constants as C


class HorseDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, labels, split, augmentation, image_size=256):
        self._image_path = image_path
        self._labels = labels
        self._transforms = get_transforms(
            split=split,
            augmentation=augmentation,
            image_size=image_size
        )

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        labels = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)
        return image, labels

class ZebraDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, labels, split, augmentation, image_size=256):
        self._image_path = image_path
        self._labels = labels
        self._transforms = get_transforms(
            split=split,
            augmentation=augmentation,
            image_size=image_size
        )

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        labels = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)
        return image, labels
