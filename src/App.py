import sys
import numpy as np
import os
import torchvision.transforms as transforms
import torch

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
from threading import Thread
from PIL import Image, ImageFilter
from torch.autograd import Variable
from src.model import Generator, SRCNN
from src.constants import *
from src.Extract import FrameExtractor
import multiprocessing as mp
from src.test_video import main
from src.BicubicInterpolation import bicubic_resize
from tqdm import tqdm
from src.FrameAssembler import assemble
import argparse

from src.test_arcnn import remove_artifacts


parser = argparse.ArgumentParser(description='Enter file path')
parser.add_argument('--path', type=str, required=True, help='path to the file you want to upscale')
parser.add_argument('--scale_factor', default=4, required=False, type=int, help='Upscale factor, choose between 2 and 4')
parser.add_argument('--arcnn', type=bool, default=False,  required=False, help='If True then remove artifacts, if False or left empty '
                                                               'then skip that step')


def enhance(x, scale, path):
    main(x, scale, path)


def bicubic(path,scale):
    bicubic_resize(path,scale)


def run():
    torch.cuda.empty_cache()
    args = parser.parse_args()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    if args.scale_factor == 2:
        model_path = MODEL_X2_DIR
    else:
        model_path = MODEL_X4_DIR

    # Extraction of frames from an original file, gets the file name and created a new save path in
    # ../enhanced_frames/filename

    x = FrameExtractor(args.path)
    x.extract()
    enhanced_dir = x.get_path()
    print(enhanced_dir)

    # Checks if a user checked ARCNN flag to TRUE.  If true, then remove artifacts
    if args.arcnn:
        print("Starting ARCNN...")
        remove_artifacts()
    print("Starting bicubic upscale...")
    p2 = mp.Process(target=bicubic, args=(EXTRACTED_FRAMES_DIR, args.scale_factor))
    p2.start()
    print("Starting SRGAN...")

    enhance(enhanced_dir, args.scale_factor, model_path)
    # implementation of multiprocessing. Frame Enhanced and bicubic upscaling run in parrarel
    # p = mp.Process(target=enhance, args=(enhanced_dir, model_path,))
    # p.start()

    # assembling frames into video
    assemble(enhanced_dir)
    assemble(BICUBIC_FRAMES_DIR)


if __name__ == '__main__':
    run()
