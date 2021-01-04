import sys
import  argparse
sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
from src.constants import *
import os
import random
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage, Normalize, RandomCrop
from PIL import Image, ImageFilter
import PIL
import numpy as np
import io
from torch.utils.data.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--out' , type=str, required=True)

def downsample():
    args = parser.parse_args()

    os.system('ffmpeg -i {} -vf scale=iw/2:-1 {}.mp4'.format(args.file,args.out))



downsample()
