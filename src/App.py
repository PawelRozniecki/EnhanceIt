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
from src.Extract import  FrameExtractor
import multiprocessing as mp
from src.test_video import main
from src.BicubicInterpolation import bicubic_resize
from tqdm import tqdm
from src.FrameAssembler import assemble
from src.test_arcnn import remove_artifacts

def arcnn():
    remove_artifacts(EXTRACTED_FRAMES_DIR)

def enhance(x):
    main(x)


def bicubic():
    bicubic_resize(ARCNN_FRAMES_DIR)


def run():
    torch.cuda.empty_cache()
    mp.set_start_method('spawn')
    path_to_video = sys.argv[1]

    x = FrameExtractor(path_to_video)
    x.extract()
    enhanced_dir = x.get_path()
    print(enhanced_dir)
    p = mp.Process(target=enhance, args=(enhanced_dir,))
    p.start()
    # p2 = mp.Process(target=bicubic)
    # p2.start()
    # p.join()
    # p2.join()
    # assemble()


if __name__ == '__main__':
    run()
