import torch
import torchvision.models as models
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from src.SRCNN.constants import *
import torchvision.transforms as transforms
from torch.autograd import Variable

from src.SRCNN.data_manager import  crop_image
from src.SRCNN.model import SRCNN
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize,Normalize, RandomHorizontalFlip

import matplotlib.pyplot as plt

def main():

    model = SRCNN()
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # image to resize

    input_image = Image.open("/Users/pingwin/PycharmProjects/EnhanceIt/src/dataset/T91/1/tt17.png").convert('RGB')
    input_image = input_image.resize(
        (
            input_image.size[0] * UPSCALE_FACTOR,
            input_image.size[1] * UPSCALE_FACTOR

        ), Image.BICUBIC
    )

    input_tensor = Variable(ToTensor()(input_image))
    input_tensor = input_tensor.view(1, -1, input_image.size[1], input_image.size[0])

    model.eval()
    output = model(input_tensor)

    output = transforms.ToPILImage()(output.data[0])
    output = output.convert('RGB')
    output.save('/Users/pingwin/PycharmProjects/EnhanceIt/src/output2.png')




if __name__ == '__main__':
    main()