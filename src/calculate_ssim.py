import cv2
from skimage.io import imread
from skimage.color import rgb2g
def SSIM(img1,img2):
    cv2.SSIM