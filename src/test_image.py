import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
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
    path = sys.argv[1]

    torch.cuda.empty_cache()
    print(DEVICE)
    filename = extract_filename(path)
    input_image = Image.open(path).convert('RGB')
    input = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)
    input = input.cuda()

    model = Generator(UPSCALE_FACTOR)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    with torch.no_grad():
        output = model(input)

    output = output.cpu()
    out_img = transforms.ToPILImage()(output[0].data.cpu())
    out_img.save(ENHANCED_IMG_DIR + filename + "_x2.png")

main()