from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage
from torch.utils import data
import os

from PIL import Image, ImageFilter

from src.constants import *


def get_images(root_dir):
    images = []
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(dir, file))

    return images

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class FolderData(data.Dataset):
    def __init__(self, root_dir, crop_size):
        super(FolderData, self).__init__()
        self.files = get_images(root_dir)

        self.input_transform = input_transform(crop_size)
        self.target_transform = get_target_transforms(crop_size)

    def __getitem__(self, index):

        prediction = load_img(self.files[index])
        target = prediction.copy()
        prediction = prediction.filter(ImageFilter.GaussianBlur(2))
        prediction = self.input_transform(prediction)
        target = self.target_transform(target)

        return prediction, target

    def __len__(self):
        return len(self.files)


def crop_image(crop_size):
    return crop_size - (crop_size % UPSCALE_FACTOR)


def get_training_set(root, crop_size, upscale_factor):
    crop_size = crop_image(crop_size)
    return FolderData(root_dir=root, crop_size=crop_size)

def get_testing_set(root,crop_size,upscale_factor):
   return FolderData(root_dir=root,  crop_size=crop_size)

def get_testing_set(root, crop_size, upscale_factor):
    return FolderData(root_dir=root, crop_size=crop_size)


def input_transform(crop_size):
    return Compose([

        CenterCrop(crop_size),
        Resize(crop_size//UPSCALE_FACTOR,interpolation=Image.BICUBIC),

        ToTensor(),

    ])


def get_target_transforms(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])

