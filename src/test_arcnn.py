import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import torch
from PIL import Image
from src.model import ARCNN
from torch.autograd import Variable
import torchvision.transforms as transforms
import io
from src.constants import *
import os
from tqdm import  tqdm


def remove_artifacts():
    path = sys.argv[1]
    model = ARCNN()

    model.load_state_dict(torch.load('/home/pawel/PycharmProjects/EnhanceIt/src/models/arcnn/ARCNN203_loss: 0.002472760060036014.pth', map_location='cpu'))
    model.eval()

    # for file in tqdm(os.listdir(path)):
    input_image = Image.open(path)
    input_image = input_image.convert('RGB')

    # buffer = io.BytesIO()
    # input_image.save(buffer, format='jpeg', quality=10)
    # input_image = Image.open(buffer)
    # input_image.save("/home/pawel/PycharmProjects/EnhanceIt/src/result/compressd_image.png")
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)

    output = output.mul(255.0).clamp(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    out_img = Image.fromarray(output, mode='RGB')
    out_img.save(ENHANCED_IMG_DIR + "arcnn.png")


remove_artifacts()

