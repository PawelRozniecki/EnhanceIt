import sys
import torchvision.transforms as transforms
import torch
import os
from PIL import Image, ImageFilter
from torch.autograd import Variable
from src.model import Generator, SRCNN
from src.constants import *
from tqdm import tqdm

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')


def main(save_path):
    torch.cuda.empty_cache()
    print(DEVICE)
    extract_path = EXTRACTED_FRAMES_DIR
    for file in tqdm(os.listdir(extract_path)):

        input_image = Image.open(extract_path+file).convert('RGB')
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
        # out_img.save('/home/pawel/PycharmProjects/EnhanceIt/src/output1.png')
        out_img.save(save_path + file)





