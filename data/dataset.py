import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from detectron2.data import DatasetMapper

from util import constants as C


class HorseDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, labels=None, transforms=None):
        self._image_path = image_path
        self._transforms = transforms
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        label = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)
        return image, labels

class ZebraDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, labels=None, transforms=None):
        self._image_path = image_path
        self._transforms = transforms
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        label = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)
        return image, labels
