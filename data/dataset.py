import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from detectron2.data import DatasetMapper
from .transforms import get_transforms

from util import constants as C

class GANDataset(torch.utils.data.Dataset):
    def __init__(self, A_image_path, B_image_path, split, augmentation, image_size=256):
        self._A = pd.read_csv(A_image_path)["image_path"]
        self._B = pd.read_csv(B_image_path)["image_path"]
        self._lenA = len(self._A)
        self._lenB = len(self._B)   

        self._transforms = get_transforms(
            split=split,
            augmentation=augmentation,
            image_size=image_size
        )

    def __getitem__(self, index):
        # We have uneven datasets for A and B
        # Eventually one of them will have to loop around
        if index >= self._lenA:
            imageA = Image.open(self._A[index - self._lenA]).convert('RGB')
            imageB = Image.open(self._B[index]).convert('RGB')
        elif index >= self._lenB:
            imageA = Image.open(self._A[index]).convert('RGB')
            imageB = Image.open(self._B[index - self._lenB]).convert('RGB')
        else:
            imageA = Image.open(self._A[index]).convert('RGB')
            imageB = Image.open(self._B[index]).convert('RGB')


        if self._transforms is not None:
            imageA = self._transforms(imageA)
            imageB = self._transforms(imageB)

        return imageA, imageB     

    def __len__(self):
        A_len = len(self._A)
        B_len = len(self._B)
        return max(A_len, B_len)



# class HorseDataset(torch.utils.data.Dataset):
#     def __init__(self, image_path, labels, split, augmentation, image_size=256):
#         self._image_path = image_path
#         self._labels = labels
#         self._transforms = get_transforms(
#             split=split,
#             augmentation=augmentation,
#             image_size=image_size
#         )

#     def __len__(self):
#         return len(self._labels)

#     def __getitem__(self, index):
#         labels = torch.tensor(np.float64(self._labels[index]))
#         image = Image.open(self._image_path[index]).convert('RGB')
#         if self._transforms is not None:
#             image = self._transforms(image)
#         return image, labels

# class ZebraDataset(torch.utils.data.Dataset):
#     def __init__(self, image_path, labels, split, augmentation, image_size=256):
#         self._image_path = image_path
#         self._labels = labels
#         self._transforms = get_transforms(
#             split=split,
#             augmentation=augmentation,
#             image_size=image_size
#         )

#     def __len__(self):
#         return len(self._labels)

#     def __getitem__(self, index):
#         labels = torch.tensor(np.float64(self._labels[index]))
#         image = Image.open(self._image_path[index]).convert('RGB')
#         if self._transforms is not None:
#             image = self._transforms(image)
#         return image, labels
