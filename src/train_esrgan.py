import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
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
from src.model import ESRGAN_Discriminator, ESRGAN, PerceptualLoss


from src.data_manager import *
from torch.utils.data.dataloader import DataLoader

from src.model import SRCNN, Generator, ESRGAN_Discriminator

torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

generator = Generator(UPSCALE_FACTOR).to(DEVICE)
discriminator = ESRGAN_Discriminator().to(DEVICE)

# dataset loaderds
train_set = TrainDatasetFromFolder(DATASET_PATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValidateDatasetFromFolder(TEST_DATAPATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)

train_loader = DataLoader(get_training_set(DATASET_PATH, SIZE, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=1)
val_loader = DataLoader(get_val_set(TEST_DATAPATH, SIZE, UPSCALE_FACTOR), batch_size=1, shuffle=False, num_workers=1)

optimizer = optim.Adam(ESRGAN.parameters(), lr=1e-4)
discriminatorOptim = optim.Adam(ESRGAN_Discriminator.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    train_bar = tqdm(train_loader)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    content_criterion = nn.L1Loss().to(DEVICE)
    perception_criterion = PerceptualLoss().to(DEVICE)
    for low_res, high_res in train_bar:
        real_data = Variable(high_res).to(DEVICE)
        batch_data = Variable(low_res).to(DEVICE)

        real_labels = torch.ones((high_res.size(0), 1)).to(DEVICE)
        fake_labels = torch.zeros((high_res.size(0), 1)).to(DEVICE)
        optimizer.zero_grad()

        fake_image = generator(low_res)

        score_real = discriminator(high_res)
        score_fake = discriminator(fake_image)

        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
        adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
        adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
        perceptual_loss = perception_criterion(high_res, fake_image)
        content_loss = content_criterion(fake_image, high_res)
        generator_loss = adversarial_loss * 0 + \
                         perceptual_loss * 0 + \
                         content_loss * 1
        generator_loss.backward()
        optimizer.step()

        # Discriminator training

        discriminatorOptim.zero_grad()
        score_real = discriminator(high_res)
        score_fake = discriminator(fake_image.detach())
        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
        adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
        discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

        discriminator_loss.backward()
        discriminatorOptim.step()

    torch.save(generator.state_dict(),
                '/home/pawel/PycharmProjects/EnhanceIt/src/models/ESRGAN/gen/cp_generator_' + str(epoch) + ".pth")
    torch.save(discriminator.state_dict(),
               '/home/pawel/PycharmProjects/EnhanceIt/src/models/ESRGAN/dis/cp_discriminator_' + str(epoch) + ".pth")




