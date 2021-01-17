import sys

sys.path.append('/Users/pingwin/PycharmProjects/EnhanceIt/')
import torch


from PIL import Image, ImageFilter
import numpy as np
from torch.autograd import Variable

import torchvision.transforms as transforms
from src.model import Generator, SRCNN
import os
from src.constants import *
from src.data_utils import extract_filename
from tqdm import tqdm


def main():

    # torch.cuda.empty_cache()
    print(DEVICE)

    path = sys.argv[1]
    input_image = Image.open(path).convert('RGB')
    input = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)
    # input = input.cuda()

    model = Generator(UPSCALE_FACTOR)
    # model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(input)

    output = output.cpu()
    out_img = transforms.ToPILImage()(output[0].data.cpu())
    out_img.save('/Users/pingwin/PycharmProjects/EnhanceIt/src/frontend/savename3.png')
    print("Done!")

main()