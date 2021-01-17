import sys

sys.path.append('/home/pawel/PycharmProjects/EnhanceIt')
import torch
import torchvision
import torch.nn as nn
from src.constants import *
from src.loss import GeneratorLoss
import matplotlib.pyplot as plt

from tqdm import tqdm
import copy
import torchvision.transforms as transforms
from torch.autograd import Variable

from src.data_utils import *
from math import log10

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader
from src.data_utils import CompressDatasetImages
from src.model import ARCNN,FastARCNN

from torchvision.datasets import ImageFolder

torch.autograd.set_detect_anomaly(True)

import torch.optim as optim

model = FastARCNN()
model = model.to(DEVICE)
print(DEVICE)
# model.load_state_dict(torch.load('/home/pawel/PycharmProjects/EnhanceIt/src/models/arcnn_checkpoints/cp499.pth', map_location=DEVICE))

print("MODEL INFORMATION\n", model)


train_set = CompressDatasetImages(ARCNN_DATASET, 10)

print(train_set.__len__())
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True, num_workers=9)
criterion = nn.MSELoss()

optimizer = optim.Adam([
    {'params': model.base.parameters(),'lr': 1e-4},
    {'params': model.last.parameters(), 'lr': 1e-4}
], lr=1e-4)

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
        out_real = transforms.ToPILImage()(inputs[0].data.cpu())
        out_real.save('/home/pawel/PycharmProjects/EnhanceIt/src/arcnn_progress/real' + str(epoch) + '.png')
        target = target.to(DEVICE)
        prediction = model(inputs)
        out_img = transforms.ToPILImage()(prediction[0].data.cpu())
        out_img.save('/home/pawel/PycharmProjects/EnhanceIt/src/arcnn_progress/compressed'+str(epoch)+'.png')

        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(train_loader)
        runtime_results['loss'] = epoch_loss
    torch.save(model.state_dict(),
               '/home/pawel/PycharmProjects/EnhanceIt/src/models/farcnn-cp/cp' + str(epoch) + ".pth")
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, epoch_loss))
    results['loss'].append(runtime_results['loss'])
    loss_arr = np.append(loss_arr, epoch_loss)
    plt.plot(epoch_arr, loss_arr, label="generator loss")
    plt.savefig('arcnn_plot.png')


torch.save(model.state_dict(), '/home/pawel/PycharmProjects/EnhanceIt/src/models/arcnn/farcnn.pth')
plt.show()
