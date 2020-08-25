import sys
sys.path.append('/Users/pingwin/PycharmProjects/EnhanceIt')
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable

import torchvision.transforms as transforms
from src.model import Generator

from src.constants import *

def main():
    torch.cuda.empty_cache()
    path = sys.argv[1]

    model = Generator(UPSCALE_FACTOR)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH,map_location='cpu'))

    model.eval()
    # image to resize
    # for file in os.listdir('/home/pawel/PycharmProjects/EnhanceIt/src/images/1'):

    input_image = Image.open(path)
    input_image = input_image.convert('RGB')
    input_image = input_image.resize((int(input_image.size[0]*UPSCALE_FACTOR), int(input_image.size[1]*UPSCALE_FACTOR)), Image.BICUBIC)
    bicubic = input_image.convert('RGB')
    bicubic.save('bicubic.png')

    input = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)

    input = input.to(DEVICE)

    with torch.no_grad():
        torch.cuda.empty_cache()
        output = model(input).clamp(0.0,1.0)

    output = output.cpu()

    out_img =transforms.ToPILImage()(output[0].data.cpu())

    out_img.save('output.png')

    print("saved sucesfully ")


main()
