import sys
sys.path.append('/Users/pingwin/PycharmProjects/EnhanceIt')

import torch
import torch.nn as nn

from tqdm import tqdm
import copy
from torch.autograd import Variable, backward


from math import log10

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader

from src.model import SRCNN, Generator, Discriminator

from torchvision.datasets import ImageFolder

import torch.optim as optim

torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = Generator(UPSCALE_FACTOR)
best_weight = copy.deepcopy(model.state_dict())
model.load_state_dict(best_weight)
discriminatorNet = Discriminator()
model.to(DEVICE)
discriminatorNet.to(DEVICE)


print("Generator INFORMATION\n", model)
print("Discriminator INFORMATION\n", discriminatorNet)
print("initiating SRCNN training... ")

# dataset = ImageFolder(DATASET_PATH, transform=transforms.ToTensor())
# train_set = get_training_set(UPSCALE_FACTOR)
# test_len = int(len(get_training_set(UPSCALE_FACTOR)) / 3)
# train_len = int(len(get_training_set(UPSCALE_FACTOR)) - test_len)
#
# train, test = data.random_split(get_training_set(UPSCALE_FACTOR), [train_len, test_len])

dataset = ImageFolder(DATASET_PATH)

data_loader = DataLoader(get_training_set(DATASET_PATH, SIZE, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_testing_set(TEST_DATAPATH,SIZE, UPSCALE_FACTOR), batch_size=1, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
discriminatorOptim = optim.Adam(discriminatorNet.parameters())

model.train()

for epoch in range(EPOCHS):

    discriminatorNet.train()

    epoch_loss = 0.0
    best_psnr = 0.0
    avg_psnr = 0.0
    best_epoch = 0

    for index, data in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()

        running_corrects = 0
        batch, target = data
        batch_size = batch.size(0)

        #-----------------------------------------#
        #--------------DISCRIMINATOR--------------#
        #-----------------------------------------#

        real_data = Variable(target).to(DEVICE)
        batch_data = Variable(batch).to(DEVICE)
        fake_data = model(batch_data)

        discriminatorNet.zero_grad()

        real_output = discriminatorNet(real_data).mean()
        fake_output = discriminatorNet(fake_data).mean()
        discriminator_loss = 1 - real_output + fake_output
        discriminator_loss.backward(retain_graph=True)
        discriminatorOptim.step()

        # -----------------------------------------#
        # ---------------GENERATOR-----------------#
        # -----------------------------------------#

        model.zero_grad()
        loss = criterion(fake_data, real_data)
        loss.backward()
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

    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss / len(data_loader)))
    # torch.save(model.state_dict(), MODEL_SAVE_PATH)

    model.eval()

    with torch.no_grad():
        for i, d in enumerate(tqdm(test_loader, desc="testing progress")):
            test_image, test_label = d[0].to(DEVICE), d[1].to(DEVICE)

            predication = model(test_image)
            loss = criterion(predication, test_label)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr

        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))

        if avg_psnr/len(test_loader) > best_psnr:
            best_epoch = epoch
            best_psnr = avg_psnr
            best_weight = copy.deepcopy(model.state_dict())
            torch.save(best_weight, MODEL_SAVE_PATH)
            print("gello")

        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr/len(test_loader)))
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        torch.save(discriminatorNet.state_dict(), '/Users/pingwin/PycharmProjects/EnhanceIt/src/models/discriminatorModel.pth')




