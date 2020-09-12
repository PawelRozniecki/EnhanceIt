import sys
sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')


import torch
import torch.nn as nn

from tqdm import tqdm
import copy

from math import log10

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader

from src.model import SRCNN, Generator

from torchvision.datasets import ImageFolder

import torch.optim as optim

torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = Generator(UPSCALE_FACTOR)
model.to(DEVICE)

print("MODEL INFORMATION\n", model)
print("initiating  training... ")

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
#
# for param in model.parameters():
#     param.requires_grad = True

best_weight = copy.deepcopy(model.state_dict())

model.train()

for epoch in range(EPOCHS):

    epoch_loss = 0.0
    best_psnr = 0.0

    for index, data in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()

        running_corrects = 0
        batch, target = data

        batch = batch.to(DEVICE)
        target = target.to(DEVICE)

        preds = model(batch)

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss / len(data_loader)))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    model.eval()
    avg_psnr = 0.0

    with torch.no_grad():
        for i, d in enumerate(tqdm(test_loader, desc="testing progress")):
            test_image, test_label = d[0].to(DEVICE), d[1].to(DEVICE)

            predication = model(test_image).clamp(0.0, 1.0)
            loss = criterion(predication, test_label)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))

        if avg_psnr > best_psnr:
            best_epoch = epoch
            best_psnr = avg_psnr
            best_weight = copy.deepcopy(model.state_dict())
            torch.save(best_weight, MODEL_SAVE_PATH)

        # print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr/len(test_loader)))
        torch.save(best_weight, '/home/pawel/PycharmProjects/EnhanceIt/src/models/bestSRGAN.pth')



