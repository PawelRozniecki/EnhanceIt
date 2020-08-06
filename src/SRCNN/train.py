import sys
sys.path.append('/Users/pingwin/PycharmProjects/EnhanceIt')


import torch
import torch.nn as nn




from PIL import Image
import PIL
from PIL import ImageFilter as IF
from tqdm import tqdm
import copy

import numpy
from math import log10
from torch.autograd import Variable

from torchvision.transforms import transforms
from src.SRCNN.constants import *
from src.SRCNN.data_manager import *
from torch.utils.data.dataloader import DataLoader

import torchvision.models as models
from src.SRCNN.model import SRCNN
import torch.nn.functional as F
from torch.utils import data

import torchvision
from torchvision.datasets import ImageFolder

import torch.optim as optim

from src.SRCNN.model import Model

torch.set_grad_enabled(True)
model = SRCNN()
model.to(DEVICE)

print("MODEL INFORMATION\n", model)
print("initiating SRCNN training... ")

# dataset = ImageFolder(DATASET_PATH, transform=transforms.ToTensor())
# train_set = get_training_set(UPSCALE_FACTOR)
# test_len = int(len(get_training_set(UPSCALE_FACTOR)) / 3)
# train_len = int(len(get_training_set(UPSCALE_FACTOR)) - test_len)
#
# train, test = data.random_split(get_training_set(UPSCALE_FACTOR), [train_len, test_len])

dataset = ImageFolder(DATASET_PATH)

data_loader = DataLoader(get_training_set(DATASET_PATH, 256, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_testing_set(DATASET_PATH, 256, UPSCALE_FACTOR), batch_size=1)

criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters()}
], lr=0.00001)




for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0.0

    best_psnr = 0.0

    best_weight = copy.deepcopy(model.state_dict())

    for index, data in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()

        running_corrects = 0
        batch, target = data

        batch = batch.to(DEVICE)
        target = target.to(DEVICE)

        preds = model(batch)

        loss = criterion(preds, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss / len(data_loader)))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    model.eval()
    avg_psnr = 0.0
    for i, d in enumerate(tqdm(test_loader, desc="testing progress")):
        test_image, test_label = d
        test_image = test_image.to(DEVICE)
        test_label = test_label.to(DEVICE)
        predication = model(test_image)
        loss = criterion(predication, test_label)
        psnr = 10 * log10(1 / loss.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))

    if (avg_psnr > best_psnr):
        best_epoch = epoch
        best_psnr = avg_psnr
        best_weight = copy.deepcopy(model.state_dict())
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr/len(test_loader)))
    torch.save(best_weight, '/home/pawel/PycharmProjects/EnhanceIt/srcbest.pth')
