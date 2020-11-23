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
from tqdm import tqdm


def main():
    torch.cuda.empty_cache()
    print(DEVICE)
    path = sys.argv[1]

    # for file in tqdm(os.listdir(path)):
    #
    #     input_image = Image.open(path+file)

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
    out_img.save('/home/pawel/PycharmProjects/EnhanceIt/src/result/out2.png')


main()
