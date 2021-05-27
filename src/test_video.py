import sys
sys.path.append('/run/timeshift/backup/thesis/EnhanceIt')

import torchvision.transforms as transforms
import torch
import os
from PIL import Image, ImageFilter
from torch.autograd import Variable
from src.model import Generator, ESRGAN_Generator
from src.constants import *
from tqdm import tqdm
from torchvision.utils import save_image

def main(frame_path, save_path, model_path):
    # print(DEVICE)
    # extract_path = EXTRACTED_FRAMES_DIR

    model = ESRGAN_Generator()
    model.load_state_dict(torch.load('/run/timeshift/backup/thesis/EnhanceIt/src/models/ESRGAN/cp_generator_528.pth'))
    model.to(DEVICE)

    for file in tqdm(os.listdir(frame_path)):
        torch.cuda.empty_cache()

        input_image = Image.open(frame_path+file).convert('RGB')
        input = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)

        input = input.cuda()

        with torch.no_grad():
            output = model(input)
        save_image(output, save_path + file)


main(EXTRACTED_FRAMES_DIR, ENHANCED_FRAMES_DIR, '/run/timeshift/backup/thesis/EnhanceIt/src/models/ESRGAN/cp_generator_528.pth')