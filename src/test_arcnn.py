import sys

sys.path.append('/run/timeshift/backup/thesis/EnhanceIt')
import torch
from PIL import Image
from src.model import ARCNN
from torch.autograd import Variable
import torchvision.transforms as transforms
import io
from src.constants import *
import os
from tqdm import  tqdm

from torchvision.utils import save_image

def remove_artifacts():
    model = ARCNN()
    model.load_state_dict(torch.load('/run/timeshift/backup/thesis/EnhanceIt/src/models/arcnn/cp35.pth'))
    # model.to(DEVICE)
    model.eval()
    path = sys.argv[1]

    # for file in tqdm(os.listdir(EXTRACTED_FRAMES_DIR)):

    input_image = Image.open(path)
    input_image = input_image.convert('RGB')
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)
    save_image(output, "arcnn2.png")

    # output = output.mul(255.0).clamp(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    # save_image(output,"arcnn1.png")
    # out_img = Image.fromarray(output, mode='RGB')
    # out_img.save("arcnn.png")
    # # out_img.save("/home/pawel/PycharmProjects/EnhanceIt/src/Single_Image_Results/arcnn1.png")


remove_artifacts()


