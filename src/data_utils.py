from src.constants import *
import os
import random
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage, Normalize, RandomCrop
from PIL import Image
import PIL
import numpy as np
import io
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter

from tqdm import tqdm


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


def crop_image(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def high_res_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor()
    ])


def low_res_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def get_training_set(root, crop_size, upscale_factor):
    crop_size = crop_image(crop_size)
    return TrainDatasetFromFolder(dataset_path=root, crop_size=crop_size, upscale_factor=upscale_factor)


def get_val_set(root, crop_size, upscale_factor):
    return ValidateDatasetFromFolder(dataset_path=root, crop_size=crop_size, upscale_factor=upscale_factor)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_path, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.files = get_images(dataset_path)
        crop_size = crop_image(crop_size, upscale_factor)
        self.hr_transform = high_res_transform(crop_size)
        self.lr_transform = low_res_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(load_img(self.files[index]))
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.files)


class ValidateDatasetFromFolder(Dataset):
    def __init__(self, dataset_path, crop_size, upscale_factor):
        super(ValidateDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size
        self.files = get_images(dataset_path)

    def __getitem__(self, index):
        hr_image = load_img(self.files[index])
        width, height = hr_image.size
        crop_size = crop_image(min(width, height), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)

        hr_restore = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore), ToTensor()(hr_image)

    def __len__(self):
        return len(self.files)


class CompressDatasetImages(Dataset):
    def __init__(self, dataset_path):
        super(CompressDatasetImages, self).__init__()
        self.files = get_images(dataset_path)
        self.dataset_path = dataset_path

    def __getitem__(self, index):
        img_to_compress = load_img(self.files[index]).convert('RGB')

        if random.random() <= 0.1:
            scale = random.choice([0.9, 0.8, 0.7, 0.6])
            img_to_compress = img_to_compress.resize((int(img_to_compress.width * scale),
                                                      int(img_to_compress.height * scale)),
                                                     resample=Image.BICUBIC)

        if random.random() <= 0.1:
            img_to_compress = img_to_compress.rotate(random.choice([90, 180, 270]), expand=True)

        crop_x = random.randint(0, img_to_compress.width - 24)

        crop_y = random.randint(0, img_to_compress.height - 24)

        img_to_compress = img_to_compress.crop((crop_x, crop_y, crop_x + 24, crop_y + 24))

        buffer = io.BytesIO()
        img_to_compress.save(buffer, format='jpeg', quality=10)
        input = Image.open(buffer)

        input = np.array(input).astype(np.float32)
        img_to_compress = np.array(img_to_compress).astype(np.float32)

        input = np.transpose(input, axes=[2, 0, 1])
        img_to_compress = np.transpose(img_to_compress, axes=[2, 0, 1])

        input /= 255
        img_to_compress /= 255

        return input, img_to_compress

    def __len__(self):
        return len(self.files)

    def compress(self):
        path = self.dataset_path
        dirs = os.listdir(path)
        count = 0
        for index, image in enumerate(tqdm(dirs)):
            img = Image.open(path + image)
            img.save("/home/pawel/PycharmProjects/EnhanceIt/src/compressed_dataset/Compressed_" + image,
                     format='jpeg',
                     Optimize=True,
                     quality=10)

    def test(self, path):
        image = Image.open(path)
        image.save("/home/pawel/PycharmProjects/EnhanceIt/src/compressed_dataset/Compressed.jpg")


if __name__ == "__main__":
    compression = CompressDatasetImages("/home/pawel/PycharmProjects/EnhanceIt/src/DIV2K/Train/X2/", 10)

    compression.compress()
