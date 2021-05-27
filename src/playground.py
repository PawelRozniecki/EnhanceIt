import sys
sys.path.append('/run/timeshift/backup/thesis/EnhanceIt')
import numpy as np
import torch

print(torch.__version__)
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage, RandomCrop, GaussianBlur,Lambda
from PIL import Image, ImageFilter
from src.data_utils import randomJPEGCompresss

# image = Image.open(
#     "/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/Single_Image_Results/manal.jpg")
#
# out = image.filter(ImageFilter.MedianFilter)
# # facto = 0.25
# # out = image.resize([int(facto * s) for s in image.size], Image.BICUBIC)
# #
#
# out.save('s.png')
l =np.array([2,2])
loss = np.arange(start=0,stop=499, step=1)
print(loss.item(2))

for epoch in range(500,1000):
    print(epoch)