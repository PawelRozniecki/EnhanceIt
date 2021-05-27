import sys
sys.path.append('/run/timeshift/backup/thesis/EnhanceIt')
import torch


from PIL import Image, ImageFilter
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

import torchvision.transforms as transforms
from src.model import Generator, SRCNN
import os
from src.constants import *
from src.benchmarks import ssim
from src.data_utils import extract_filename
from tqdm import tqdm
from  torchvision.transforms import  ToTensor, ToPILImage
from src.BicubicInterpolation import bicubic_resize

def main():
    print(DEVICE)
    path = sys.argv[1]
    # bicubic_resize(path)
    torch.cuda.empty_cache()
    print(DEVICE)
    filename = extract_filename(path)
    input_image = Image.open(path).convert('RGB')
    original_image = Image.open(path)
    # bicubic = Image.open("/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/Single_Image_Results/bicubic.png")

    hr = Variable(transforms.ToTensor()(original_image).unsqueeze(0))
    # bic = Variable(transforms.ToTensor()(bicubic).unsqueeze(0))
    lr = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)
    model = Generator(4).eval()
    model.load_state_dict(torch.load('/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/SRGAN_with_blur/models/cp362.pth',map_location='cpu'))

    # model.to(DEVICE)
    with torch.no_grad():
        sr = model(lr)
        save_image(sr, ENHANCED_IMG_DIR + filename + "z362ebra_x4.png")


main()
