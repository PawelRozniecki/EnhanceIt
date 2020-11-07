import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import torch
import torchvision
import torch.nn as nn
from src.constants import *
from src.loss import GeneratorLoss

from tqdm import tqdm
import copy
import torchvision.transforms as transforms
from torch.autograd import Variable

from src.data_utils import *
from math import log10

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader
from src.data_utils import CompressDatasetImages
from src.model import ARCNN

from torchvision.datasets import ImageFolder

torch.autograd.set_detect_anomaly(True)

import torch.optim as optim

model = ARCNN()
model = model.to(DEVICE)
model.load_state_dict(torch.load('/home/pawel/PycharmProjects/EnhanceIt/src/models/ARCNN.pth', map_location=DEVICE))

print("MODEL INFORMATION\n", model)


train_set = CompressDatasetImages('/home/pawel/Downloads/images_final/Twitter_Compressed','/home/pawel/Downloads/images_final/Original')
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=4)


criterion = nn.MSELoss()

optimizer = optim.Adam([
    {'params': model.base.parameters()},
    {'params': model.last.parameters(), 'lr': 5e-4*0.1}
], lr=5e-4)

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    model.train()

    for index, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        inputs, target = data
        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)
        prediction = model(inputs)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))
    torch.save(model.state_dict(), '/home/pawel/PycharmProjects/EnhanceIt/src/models/ARCNN.pth')
torch.save(model.state_dict(), '/home/pawel/PycharmProjects/EnhanceIt/src/models/ARCNN.pth')
