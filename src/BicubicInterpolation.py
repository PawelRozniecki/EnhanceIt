import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
from PIL import Image, ImageFilter
from src.constants import *
from tqdm import tqdm
import os


def bicubic_resize(path,scale):
    for file in tqdm(os.listdir(path)):
        input_image = Image.open(path + file).convert('RGB')
        input_image = input_image.resize((int(input_image.width * scale),
                                          int(input_image.height * scale)),
                                         resample=Image.BICUBIC)

        input_image.save(BICUBIC_FRAMES_DIR + file)
