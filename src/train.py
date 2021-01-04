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

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader

from src.model import SRCNN, Generator, Discriminator

torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

generator = Generator(UPSCALE_FACTOR)

discriminatorNet = Discriminator()
# uncomment to load model from checkpoint
generator.load_state_dict(torch.load('/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/cp358.pth', map_location=DEVICE))
discriminatorNet.state_dict(torch.load('/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/dis32px.pth'))


print("Generator INFORMATION\n", generator)
print("Discriminator INFORMATION\n", discriminatorNet)
print("initiating SRCNN training... ")

# dataset loaderds
train_set = TrainDatasetFromFolder(DATASET_PATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValidateDatasetFromFolder(TEST_DATAPATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)

train_loader = DataLoader(get_training_set(DATASET_PATH, SIZE, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=1)
val_loader = DataLoader(get_val_set(TEST_DATAPATH, SIZE, UPSCALE_FACTOR), batch_size=1, shuffle=False, num_workers=1)

criterion = GeneratorLoss()

generator.cuda()
discriminatorNet.cuda()
criterion.cuda()

# optimizers
optimizer = optim.Adam(generator.parameters(), lr=1e-4)
discriminatorOptim = optim.Adam(discriminatorNet.parameters(), lr=1e-4)

best_epoch = 0
best_psnr = 0.0
epoch_arr = np.empty(0)
loss_arr = np.empty(0)
gscr_arr = np.empty(0)
dscr_arr = np.empty(0)

results = {'gen_loss': [], 'dis_loss': [], 'd_score': [], 'g_score': [], 'psnr': []}

for epoch in range(EPOCHS):
    epoch_arr = np.append(epoch_arr, epoch)
    generator_loss = 0.0
    avg_psnr = 0.0
    loss = 0
    discriminator_loss = 0
    runtime_results = {'gen_loss': 0, 'dis_loss': 0, 'd_score': 0, 'g_score': 0}
    train_bar = tqdm(train_loader)
    generator.train()
    discriminatorNet.train()

    for data, target in train_bar:
        batch_size = data.size(0)
        # -----------------------------------------#
        # --------------DISCRIMINATOR--------------#
        # -----------------------------------------#

        real_data = Variable(target).to(DEVICE)
        out_real = transforms.ToPILImage()(real_data[0].data.cpu())
        out_real.save('/home/pawel/PycharmProjects/EnhanceIt/src/result/real' + str(epoch) + '.png')
        batch_data = Variable(data).to(DEVICE)

        fake_data = generator(batch_data)
        out_img = transforms.ToPILImage()(fake_data[0].data.cpu())
        out_img.save('/home/pawel/PycharmProjects/EnhanceIt/src/result/fake' + str(epoch) + '.png')
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
        fake_img = generator(batch_data)
        restored = transforms.ToPILImage()(fake_img[0].data.cpu())
        restored.save('/home/pawel/PycharmProjects/EnhanceIt/src/result/restored' + str(epoch) + '.png')
        optimizer.step()

        generator_loss += loss.item() / len(train_loader)
        runtime_results['gen_loss'] = generator_loss
        runtime_results['dis_loss'] = discriminator_loss.item()
        runtime_results['d_score'] = real_output.item()
        runtime_results['g_score'] = fake_output.item()

    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, generator_loss))
    # torch.save(generator.state_dict(), MODEL_SAVE_PATH)

    generator.eval()
    discriminatorNet.eval()

    with torch.no_grad():
        validation_results = {'psnr': 0, 'mse': 0}
        for i, d in enumerate(tqdm(val_loader, desc="testing progress")):
            test_image, test_label = d[0].to(DEVICE), d[1].to(DEVICE)

            predication = generator(test_image)
            loss = criterion(predication, test_label)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr

        final_psnr = avg_psnr / len(val_loader)
        print("===> Avg. PSNR: {:.4f} dB".format(final_psnr))
        validation_results['psnr'] = final_psnr

        torch.save(generator.state_dict(),
                   '/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/cp' + str(epoch) + ".pth")

        if avg_psnr / len(val_loader) > best_psnr:
            best_epoch = epoch
            best_psnr = avg_psnr / len(val_loader)
            best_weight = copy.deepcopy(generator.state_dict())
            torch.save(best_weight, MODEL_SAVE_PATH)

            print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
            torch.save(generator.state_dict(), MODEL_SAVE_PATH)
            torch.save(discriminatorNet.state_dict(),
                       DIS_PATH)

        results['dis_loss'].append(runtime_results['dis_loss'])
        results['gen_loss'].append(runtime_results['gen_loss'])
        results['psnr'].append(validation_results['psnr'])
        results['d_score'].append(runtime_results['d_score'])
        results['g_score'].append(runtime_results['g_score'])
        loss_arr = np.append(loss_arr, generator_loss)
        dscr_arr = np.append(dscr_arr, runtime_results['d_score'])
        gscr_arr = np.append(gscr_arr, runtime_results['g_score'])


        csv_path = '/home/pawel/PycharmProjects/EnhanceIt/src/statistics/'
        data_frame = pd.DataFrame(

            data={'D_Loss': results['dis_loss'], 'G_Loss': results['gen_loss'], 'G_Score': results['g_score'],
                  'D_Score': results['d_score'], 'PSNR': results['psnr']},
            index=range(0, epoch + 1))
        data_frame.to_csv(csv_path + 'training_results.csv', index_label='Epoch')

        plt.subplot(2, 1, 1)
        plt.plot(epoch_arr, loss_arr, label="generator loss")
        plt.subplot(2, 1, 2)
        plt.plot(epoch_arr, dscr_arr, label="discriminator score ")
        plt.plot(epoch_arr, gscr_arr, label="generator score")
        plt.savefig('plots.png')
plt.show()


