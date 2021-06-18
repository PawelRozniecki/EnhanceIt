import sys
import cv2
import sys
import os
import torch
from tqdm import tqdm
import argparse
from model import ESRGAN_Generator, ARCNN, EsrganGenerator
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from  torchvision.transforms import  ToTensor, ToPILImage
from constants import *
import  numpy as np
def extract_filename(path):

    base = os.path.basename(path)
    filename = os.path.splitext(base)[0]
    return filename

parser = argparse.ArgumentParser(description='Enter file path')
parser.add_argument('--path', type=str, required=True, help='path to the file you want to upscale')
parser.add_argument('--arcnn', type=int, default=0,  required=True, help='If True then remove artifacts, if False or left empty '
                    
                                                               'then skip that step')


def main():
    args = parser.parse_args()

    arcnn = ARCNN()
    arcnn.load_state_dict(
        torch.load("/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/experiment_models/ARCNN_ON_DIV2k/models/cp499.pth"))
    arcnn = arcnn.to(DEVICE)

    arcnn.eval()

    model = EsrganGenerator()
    model.load_state_dict(torch.load('/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/Single_Image_Results/ESRGAN16_DF2K-a03a643d.pth',  map_location='cpu'))
    # model.load_state_dict(torch.load('/home/pawel/PycharmProjects/pythonProject/models/generator_499.pth'))
    model.eval()
    model.to(DEVICE)

    if args.path is not None:

        filename = os.path.basename(args.path)
        sr_writer_path = os.path.join("sr_videos", f"sr_{4}x_{filename}")
        print(sr_writer_path)
        video_capture = cv2.VideoCapture(args.path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        lr_frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        sr_frame_size =(lr_frame_size[0] * 4, lr_frame_size[1] * 4)
        # creates VideoWriter object
        sr_writer = cv2.VideoWriter(sr_writer_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_frame_size)
        success, image = video_capture.read()

        for _ in tqdm(range(total), desc="[Super resolution process started]"):

            torch.cuda.empty_cache()

            if success:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                input = Variable(transforms.ToTensor()(img)).unsqueeze(0)
                input = input.cuda()
                if args.arcnn == 0:
                    with torch.no_grad():
                        sr = model(input)
                        sr = sr.squeeze()
                        sr = sr.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                        # Save frames to video file
                        sr_writer.write(sr)
                        # cv2.imshow("SR progress", sr)
                        # cv2.waitKey(0)
                        torch.cuda.empty_cache()

                if args.arcnn == 1:

                    with torch.no_grad():
                        a = arcnn(input)
                        sr = model(a)
                        sr = sr.squeeze()
                        sr = sr.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                        # Save frames to video file
                        sr_writer.write(sr)


            success, image = video_capture.read()

main()


