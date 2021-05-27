import cv2
import torch
from torchvision import transforms




def psnr(img1, img2):
 return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

if __name__ == '__main__':
    im1 = cv2.imread("/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/zebra.png")
    im2 = cv2.imread("/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/zebra copy.png")
    im3 = cv2.imread("/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/grames3765.png")

    print(cv2.PSNR(im1,im2))

    t = transforms.Compose([transforms.ToTensor()])
    tenso1 = t(im1)
    tenso2 = t(im2)
    tenso3 = t(im3)
    print(psnr(tenso1,tenso2))
    print(psnr(tenso1,tenso3))


