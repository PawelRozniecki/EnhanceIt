import sys
sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import torch
torch.cuda.empty_cache()
from PIL import Image
import numpy as np
from torch.autograd import Variable

import torchvision.transforms as transforms
from src.model import Generator, SRCNN

from src.constants import *

def main():
    torch.cuda.empty_cache()
    path = sys.argv[1]

    model = Generator(UPSCALE_FACTOR)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    model.eval()
    # image to resize
    # for file in os.listdir('/home/pawel/PycharmProjects/EnhanceIt/src/images/1'):

    input_image = Image.open(path)
    # input_image = input_image.convert('YCbCr')
    # input_image = input_image.resize((int(input_image.size[0]*UPSCALE_FACTOR), int(input_image.size[1]*UPSCALE_FACTOR)), Image.BICUBIC)
    # bicubic = input_image.convert('RGB')
    # bicubic.save('bicubic.png')

    input = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)
    # y, cb, cr = input_image.split()
    #
    # input_tensor = transforms.ToTensor()
    # input = input_tensor(y).view(1, -1, input_image.size[1], input_image.size[0])
    input = input.to(DEVICE)

    with torch.no_grad():
        output = model(input)

    out_img = transforms.ToPILImage()(output[0].data.cpu())

    # output = output.cpu()
    #
    # out_img_y = output[0].detach().numpy()
    # out_img_y *= 255.0
    # out_img_y = out_img_y.clip(0, 255)
    # out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    # out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    out_img.save('output.png')

    print("saved sucesfully ")


main()
