import torch.nn as nn
import torchvision.transforms as T

def get_transforms(split, augmentation, image_size, pretrained):
    
    if split != "train":
        augmentation = 'none' # Only do augmentations if its the training dataset
        
    augmentation_transforms = { # 4 different levels of augmentation transformations
        'none': [nn.Identity()],
        'flip': [T.RandomVerticalFlip(), T.RandomHorizontalFlip()],
        'affine': [T.RandomVerticalFlip(), 
                   T.RandomHorizontalFlip(),
                   T.RandomAffine(degrees=90, translate=(0.03, 0.03), scale=(0.95, 1.05))],
        'aggressive': [T.RandomResizedCrop(image_size, scale=(0.9, 1.1), ratio=(1.0, 1.0)),
                       T.RandomVerticalFlip(),
                       T.RandomHorizontalFlip(),
                       T.RandomAffine(degrees=90, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                       T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)]
    }
        
    augmentation_transform = augmentation_transforms[augmentation] 
    resize_transform = [T.Resize((image_size, image_size))] # Resize to square of specified dimensions
    totensor_transform = [T.ToTensor()] # Convert to tensor
    normalize_transform = T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # Normalize

    transforms_list = augmentation_transform + resize_transform + totensor_transform
    #if pretrained: # Normalized based on ImageNet if we are using a pre-trained model
        #transforms_list.append(normalize_transform) 
        
    transforms = T.Compose(transforms_list)
    return transforms