from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize,Normalize, RandomHorizontalFlip
from torchvision.datasets import ImageFolder
from torch.utils import data
import os

from PIL import Image, ImageFilter

from src.SRCNN.constants import *


def get_images(root_dir):

    images = []
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(dir, file))

    return images


class FolderData(data.Dataset):
    def __init__(self, root_dir, crop_size):
        super(FolderData,self).__init__()
        self.files = get_images(root_dir)
        self.input_transform = input_transform(crop_size)
        self.target_transform = get_target_transforms(crop_size)


    def __getitem__(self, index):

        pred = Image.open(self.files[index]).convert('RGB')
        target = pred.copy()
        pred = pred.filter(ImageFilter.GaussianBlur(2))

        pred = self.input_transform(pred)
        target = self.target_transform(target)

        return pred, target

    def __len__(self):
        return len(self.files)


def crop_image(crop_size):
    return crop_size - (crop_size % UPSCALE_FACTOR)


def get_training_set(root, crop_size, upscale_factor):
    crop_size = crop_image(crop_size)
    return FolderData(root_dir=root, crop_size=crop_size)

def get_testing_set(root,crop_size,upscale_factor):
   return FolderData(root_dir=root,  crop_size=crop_size)

def input_transform(crop_size):
    return Compose([

        CenterCrop(crop_size),
        Resize(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),

        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    ])

def get_target_transforms(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])

