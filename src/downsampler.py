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

parser = argparse.ArgumentParser(description='Enter file name output name and output width')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--out' , type=str, required=True)
parser.add_argument('--w', type=int)

def downsample():
    args = parser.parse_args()

    os.system('ffmpeg -i {} -vf scale={}:-1 {}.mp4'.format(args.file,args.w,args.out))



downsample()
