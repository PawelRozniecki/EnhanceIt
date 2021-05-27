import sys
sys.path.append('/Users/pingwin/PycharmProjects/EnhanceIt')
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage, Normalize
from torch.utils import data
import os
from PIL import Image, ImageFilter
from src.constants import *
from src.data_utils import *

import torch
import numpy as np


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

        # prediction = prediction.filter(ImageFilter.GaussianBlur(2))

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




def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        Lambda(randomJPEGCompresss),
        Resize(crop_size//UPSCALE_FACTOR, interpolation=Image.BICUBIC),
        ToTensor(),
    ])

def get_target_transforms(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])

