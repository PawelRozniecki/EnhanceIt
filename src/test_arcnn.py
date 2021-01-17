import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import torch
from PIL import Image
from src.model import ARCNN,FastARCNN
from torch.autograd import Variable
import torchvision.transforms as transforms
import io
from src.constants import *
import os
from tqdm import  tqdm


def remove_artifacts():
    model = ARCNN()
    model.load_state_dict(torch.load(ARCNN_MODEL))
    model.to(DEVICE)
    model.eval()

    for file in tqdm(os.listdir(EXTRACTED_FRAMES_DIR)):

        input_image = Image.open(EXTRACTED_FRAMES_DIR+file)
        input_image = input_image.convert('RGB')
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_image)

        output = output.mul(255.0).clamp(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        out_img = Image.fromarray(output, mode='RGB')
        out_img.save(ARCNN_FRAMES_DIR + file)
        # out_img.save("/home/pawel/PycharmProjects/EnhanceIt/src/Single_Image_Results/arcnn1.png")


remove_artifacts()


