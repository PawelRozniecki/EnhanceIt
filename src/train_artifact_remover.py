import sys

sys.path.append('/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt')
import torch
import torchvision
import torch.nn as nn
from src.constants import *
from src.loss import GeneratorLoss
import matplotlib.pyplot as plt
from torchvision.utils import save_image

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
print(DEVICE)
model.load_state_dict(
    torch.load('/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/models/arcnn/cp12.pth'))

print("MODEL INFORMATION\n", model)


train_set = CompressDatasetImages('/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/Datasets/flickr/val', 10)

print(train_set.__len__())
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
criterion = nn.MSELoss()

optimizer = optim.Adam([
    {'params': model.base.parameters(),'lr':  5e-4},
    {'params': model.last.parameters(), 'lr': 5e-4 * 0.1}
], lr=5e-4)

results = {'loss': [], 'psnr': []}
epoch_arr = np.empty(0)
loss_arr = np.empty(0)

for epoch in range(EPOCHS):

    epoch_arr = np.append(epoch_arr, epoch)

    epoch_loss = 0.0
    runtime_results = {'loss': 0, 'psnr': 0}

    model.train()

    for index, data in enumerate(tqdm(train_loader)):
        inputs, target = data
        inputs = inputs.to(DEVICE)
        save_image(inputs,'/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/arcnn_progress/real' + str(epoch) + '.png')

        target = target.to(DEVICE)
        prediction = model(inputs)
        save_image(prediction,'/run/media/pawel/2ad703a8-67bd-4448-a48f-28efca5d9ca0/thesis/EnhanceIt/src/arcnn_progress/compressed'+str(epoch)+'.png')

        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(train_loader)
        runtime_results['loss'] = epoch_loss
    torch.save(model.state_dict(),
               '/home/pawel/PycharmProjects/EnhanceIt/src/models/arcnn/cp' + str(epoch) + ".pth")
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss))
    results['loss'].append(runtime_results['loss'])
    loss_arr = np.append(loss_arr, epoch_loss)
    plt.plot(epoch_arr, loss_arr, label="generator loss")
    plt.savefig('arcnn_plot.png')


torch.save(model.state_dict(), '/home/pawel/PycharmProjects/EnhanceIt/src/models/arcnn/finalArcnn.pth')
plt.show()
