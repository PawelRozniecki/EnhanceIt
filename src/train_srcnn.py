import sys

from torchvision.datasets import ImageFolder

sys.path.append('/run/timeshift/backup/thesis/EnhanceIt')
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from src.constants import *
import pandas as pd
from src.loss import GeneratorLoss

from tqdm import tqdm
import copy
from torch.autograd import Variable

from src.data_utils import *
from math import log10

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader

from src.model import SRCNN, Generator, Discriminator
from src.data_manager import *
torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

torch.set_grad_enabled(True)
model = SRCNN().to(DEVICE)
print(DEVICE)
model = SRCNN()
model.to(DEVICE)

print("MODEL INFORMATION\n", model)
print("initiating SRCNN training... ")

#
# train, test = data.random_split(get_training_set(UPSCALE_FACTOR), [train_len, test_len])

dataset = ImageFolder(DATASET_PATH)

data_loader = DataLoader(get_training_set(DATASET_PATH, 256, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_testing_set(DATASET_PATH, 256, UPSCALE_FACTOR), batch_size=1)

criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.conv3.parameters()}
], lr=0.00001)




for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0.0
    avg_psnr = 0.0

    best_psnr = 0.0

    best_weight = copy.deepcopy(model.state_dict())

    running_corrects = 0
    batch, target = data

    batch.to(DEVICE)
    target.to(DEVICE)
    batch = batch.to(DEVICE)
    target = target.to(DEVICE)

    preds = model(batch)

    loss = criterion(preds, target)
    epoch_loss += loss.item()
    loss.backward()
    optimizer.step()
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss/len(data_loader)))
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss / len(data_loader)))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # model.eval()


    # for i, d in enumerate(tqdm(test_loader, desc="testing progress")):
    #
    #     test_image, test_label = d
    #     test_image = test_image.to(DEVICE)
    #     test_label = test_label.to(DEVICE)
    #     predication = model(test_image)
    #     loss = loss_func(predication, test_label)
    #     psnr = 10 * log10(1 / loss.data[0])
    #     avg_psnr += psnr
    # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))
    #
    # if (avg_psnr > best_psnr):
    #
    #     best_epoch = epoch
    #     best_psnr = avg_psnr
    #     best_weight = copy.deepcopy(model.state_dict())
    # print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    # torch.save(best_weight, '/Users/pingwin/PycharmProjects/EnhanceIt/src/best.pth')
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