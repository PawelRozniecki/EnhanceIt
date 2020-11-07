import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import torch
import torchvision
import torch.nn as nn
from src.constants import *
from src.loss import GeneratorLoss

from tqdm import tqdm
import copy
from torch.autograd import Variable

from src.data_utils import *
from math import log10

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader

from src.model import SRCNN, Generator, Discriminator, FeatureExtractor

from torchvision.datasets import ImageFolder

torch.autograd.set_detect_anomaly(True)

import torch.optim as optim

generator = Generator(UPSCALE_FACTOR)
# best_weight = copy.deepcopy(generator.state_dict())
# generator.load_state_dict(best_weight)
discriminatorNet = Discriminator()

generator.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))


print("Generator INFORMATION\n", generator)
print("Discriminator INFORMATION\n", discriminatorNet)
print("initiating SRCNN training... ")

# dataset = ImageFolder(DATASET_PATH, transform=transforms.ToTensor())
# train_set = get_training_set(UPSCALE_FACTOR)
# test_len = int(len(get_training_set(UPSCALE_FACTOR)) / 3)
# train_len = int(len(get_training_set(UPSCALE_FACTOR)) - test_len)
#
# train, train = data.random_split(get_training_set(UPSCALE_FACTOR), [train_len, test_len])
#
# dataset = ImageFolder(DATASET_PATH)
#
# data_loader = DataLoader(get_training_set(DATASET_PATH, SIZE, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(get_testing_set(TEST_DATAPATH,SIZE, UPSCALE_FACTOR), batch_size=1, shuffle=False)


# dataset loaderds
train_set = TrainDatasetFromFolder(DATASET_PATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValidateDatasetFromFolder(TEST_DATAPATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)

train_loader = DataLoader(get_training_set(DATASET_PATH, SIZE, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
val_loader = DataLoader(get_val_set(TEST_DATAPATH, SIZE, UPSCALE_FACTOR), batch_size=1, shuffle=False,num_workers=8)

criterion = GeneratorLoss()

vgg_loss = FeatureExtractor()
print(vgg_loss)

generator.to(DEVICE)
discriminatorNet.to(DEVICE)
criterion.to(DEVICE)
vgg_loss.to(DEVICE)
# optimizers
optimizer = optim.Adam(generator.parameters(), lr=1e-4)
discriminatorOptim = optim.Adam(discriminatorNet.parameters(), lr=1e-4)

best_epoch = 0
best_psnr = 0.0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    avg_psnr = 0.0
    loss = 0
    discriminator_loss = 0
    train_bar = tqdm(train_loader)
    generator.train()
    discriminatorNet.train()

    for data, target in train_bar:
        batch_size = data.size(0)
        # -----------------------------------------#
        # --------------DISCRIMINATOR--------------#
        # -----------------------------------------#

        real_data = Variable(target).to(DEVICE)

        batch_data = Variable(data).to(DEVICE)

        fake_data = generator(batch_data)

        discriminatorNet.zero_grad()

        real_output = discriminatorNet(real_data).mean()
        fake_output = discriminatorNet(fake_data).mean()
        discriminator_loss = 1 - real_output + fake_output
        discriminator_loss = discriminator_loss.mean()
        discriminator_loss.backward(retain_graph=True)
        discriminatorOptim.step()




        # -----------------------------------------#
        # ---------------GENERATOR-----------------#
        # -----------------------------------------#

        generator.zero_grad()

        loss = criterion(fake_data, real_data)

        loss.backward()

        # fake_img = generator(batch_data)
        # fake_output = discriminatorNet(fake_img).mean()

        fake_img = generator(fake_data)
        fake_output = discriminatorNet(fake_img).mean()
        optimizer.step()

        # batch = batch.to(DEVICE)
        # target = target.to(DEVICE)
        #
        # preds = model(batch)
        #
        # loss = criterion(preds, target)
        # loss.backward()
        # optimizer.step()
        epoch_loss += loss.item()
    #
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))
    # torch.save(generator.state_dict(), MODEL_SAVE_PATH)

    generator.eval()
    discriminatorNet.eval()

    with torch.no_grad():
        for i, d in enumerate(tqdm(val_loader, desc="testing progress")):
            test_image, test_label = d[0].to(DEVICE), d[1].to(DEVICE)

            predication = generator(test_image)
            loss = criterion(predication, test_label)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr

        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_loader)))

        if avg_psnr / len(val_loader) > best_psnr:
            best_epoch = epoch
            best_psnr = avg_psnr / len(val_loader)
            best_weight = copy.deepcopy(generator.state_dict())
            torch.save(best_weight, MODEL_SAVE_PATH)

            print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
            torch.save(generator.state_dict(), MODEL_SAVE_PATH)
            torch.save(discriminatorNet.state_dict(),
                       '/home/pawel/PycharmProjects/EnhanceIt/src/models/discriminatorModel.pth')


