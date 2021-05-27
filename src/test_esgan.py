import sys
sys.path.append('/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/')
import torch


from PIL import Image, ImageFilter
import numpy as np
from torch.autograd import Variable

import torchvision.transforms as transforms
from src.model import ESRGAN_Generator, SRCNN
import os
from torchvision.utils import save_image
from src.constants import *
from src.benchmarks import ssim
from src.data_utils import extract_filename
from tqdm import tqdm
from  torchvision.transforms import  ToTensor, ToPILImage
from src.BicubicInterpolation import bicubic_resize

def main():

    path = sys.argv[1]
    input_image = Image.open(path).convert("RGB")
    lr = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)
    lr = lr.cuda()

    model = ESRGAN_Generator().eval()
    # ESRGANsmallerblurLR0001 cp 51,82 ,108 143 200 235 485 the best
    model.load_state_dict(torch.load('/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/experiment_models/ESRGANsmallerblurLR0001/trainedWithFaces/generator_805.pth'))
    model.to(DEVICE)

    print(DEVICE)
    filename = extract_filename(path)
    model.eval()
    with torch.no_grad():
        sr = model(lr)
    save_image(sr, "grames3765.png")
        # out_img = transforms.ToPILImage()(sr[0].data.cpu())
        # out_img.save(ENHANCED_IMG_DIR + filename + "esrgan.png")


main()
