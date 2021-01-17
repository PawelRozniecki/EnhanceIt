import sys
sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')

import torchvision.transforms as transforms
import torch
import os
from PIL import Image, ImageFilter
from torch.autograd import Variable
from src.model import Generator
from src.constants import *
from tqdm import tqdm



def main(save_path, scale, model_path):
    torch.cuda.empty_cache()
    # print(DEVICE)
    extract_path = EXTRACTED_FRAMES_DIR
    for file in tqdm(os.listdir(extract_path)):

        input_image = Image.open(extract_path+file).convert('RGB')

        input = Variable(transforms.ToTensor()(input_image)).unsqueeze(0)

        input = input.to(DEVICE)

        model = Generator(scale)
        model.to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            output = model(input)

        output = output.cpu()
        out_img = transforms.ToPILImage()(output[0].data.cpu())
        # out_img.save('/home/pawel/PycharmProjects/EnhanceIt/src/output1.png')
        out_img.save(save_path + file)

