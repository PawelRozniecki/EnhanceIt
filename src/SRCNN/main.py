import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import os
import torch
import torchvision.models as models
from PIL import Image, ImageEnhance, ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from src.SRCNN.constants import *
import torchvision.transforms as transforms
from torch.autograd import Variable

from src.SRCNN.data_manager import crop_image
from src.SRCNN.model import SRCNN
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize, RandomHorizontalFlip


def main():
    path = sys.argv[1]

    model = SRCNN()
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # image to resize

    input_image = Image.open(path)
    input_image = input_image.convert('YCbCr')
    input_image = input_image.resize((int(input_image.size[0]*UPSCALE_FACTOR), int(input_image.size[1]*UPSCALE_FACTOR)), Image.BICUBIC)
    bicubic = input_image.convert('RGB')
    bicubic.save('/home/pawel/PycharmProjects/EnhanceIt/src/bicubic.png')

    model.eval()
    y, cb, cr = input_image.split()

    input_tensor = transforms.ToTensor()
    input = input_tensor(y).view(1, -1, input_image.size[1], input_image.size[0])

    input = input.to(DEVICE)

    with torch.no_grad():
        output = model(input).clamp(0.0,1.0)

    output = output.cpu()

    out_img_y = output[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    out_img.save('/home/pawel/PycharmProjects/EnhanceIt/src/output.png')

    print("saved sucesfully ")


main()
