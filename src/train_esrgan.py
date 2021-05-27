import sys

sys.path.append('/run/timeshift/backup/thesis/EnhanceIt')
import torch
import lpips
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from src.ESRGAN_Loss import *
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.ticker

from src.data_utils import *

from src.data_manager import *
from torch.utils.data.dataloader import DataLoader
from  src.constants import *
from src.model import ESRGAN_Discriminator,ESRGAN_Generator

torch.autograd.set_detect_anomaly(True)

import torch.optim as optim


generator = ESRGAN_Generator()
discriminator = ESRGAN_Discriminator()
generator.load_state_dict(torch.load('/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/ESRGANsmallerblurLR0001/generator_499.pth'))
discriminator.state_dict(torch.load('/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/ESRGANsmallerblurLR0001/discriminator_499.pth'))
generator.cuda()
discriminator.cuda()
perceptual_criterion = VGGLoss().to(DEVICE)
content_criterion = nn.L1Loss().to(DEVICE)
adversarial_criterion = nn.BCEWithLogitsLoss().to(DEVICE)
# psnr_loss = nn.MSELoss.to(DEVICE)

train_set = TrainDatasetFromFolder(DATASET_PATH, crop_size=SIZE, upscale_factor=UPSCALE_FACTOR)

train_loader = DataLoader(get_training_set(DATASET_PATH, SIZE, UPSCALE_FACTOR), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2)

optimizer = optim.Adam(generator.parameters(), lr=2e-4)

discriminatorOptim = optim.Adam(discriminator.parameters(), lr=2e-4)

best_psnr = 0

generator.train()
discriminator.train()
loss_arr =np.array([1.69613198,1.45448881,1.39629861,1.39314301,1.40580382,1.38206209
,1.38097366,1.3646387,1.38899444,1.35443463,1.34287236,1.3549109
,1.3315572,1.34413832,1.3600734,1.31126988,1.29065259,1.30351817
,1.30439034,1.30249788,1.29959251,1.28013567,1.30307891,1.29454367
,1.28657978,1.28428616,1.27868414,1.26857318,1.26462175,1.25330557
,1.2736503,1.24145796,1.2833652,1.26896153,1.25990675,1.24904553
,1.2498997,1.2557406,1.22483051,1.23621377,1.23469733,1.25389409
,1.1954296,1.23241514,1.23909966,1.20713071,1.21026663,1.20475603
,1.22518118,1.19792833,1.20163474,1.1920351,1.20581293,1.18236112
,1.18964163,1.18774066,1.15439112,1.14149209,1.15623072,1.14820069
,1.13257817,1.13400034,1.11972329,1.12691208,1.07668106,1.07526391
,1.07890067,1.08396086,1.06758803,1.07524392,1.07620706,1.08552778
,1.06118582,1.04982358,1.02891636,1.03179267,1.05233388,1.00781009
,0.99723687,1.00265619,1.01370738,1.0359953,0.97813073,0.98990657
,0.99949915,1.00545151,0.9884951,0.9796185,1.01412269,0.95124429
,0.95986968,0.94873629,0.96275997,0.96214776,0.97442635,0.95579401
,0.96693642,0.92855412,0.9409889,0.96310273,0.92970135,0.92529419
,0.94193358,0.91732893,0.91505847,0.9054089,0.885634,0.88505724
,0.88055925,0.89106943,0.91077732,0.89515648,0.89378825,0.88786634
,0.88948077,0.89697957,0.86894451,0.89124897,0.86441725,0.85615893
,0.84273712,0.86443163,0.85275498,0.865938,0.85066317,0.86518939
,0.85032425,0.8325418,0.84350768,0.86790238,0.86276122,0.83678651
,0.83055222,0.82544426,0.8126826,0.78980232,0.80806793,0.8121148
,0.80060237,0.80528261,0.80795378,0.81229989,0.81631148,0.80298033
,0.81326038,0.79321384,0.79192302,0.79407524,0.81004589,0.78665326
,0.78685607,0.80219251,0.79796111,0.78216731,0.80032143,0.78240843
,0.78263617,0.77792816,0.76039419,0.78583693,0.76655228,0.763049
,0.75623147,0.77411382,0.76888333,0.76087428,0.7820373,0.78815604
,0.76293637,0.76079862,0.76317962,0.78928272,0.75167329,0.74478751
,0.76937008,0.76168673,0.74191555,0.73486275,0.75003492,0.76130307
,0.74748296,0.74873772,0.75173912,0.75146598,0.73605543,0.75566953
,0.75462804,0.77259345,0.75347795,0.75031838,0.72861745,0.74908365
,0.75243143,0.74795708,0.73356032,0.73145779,0.745728,0.73491929
,0.752528,0.73727156,0.72755104,0.72560564,0.72995615,0.72567144
,0.7250976,0.7287709,0.71778144,0.7343596,0.71236545,0.71158005
,0.72998828,0.70960532,0.70426535,0.7274479,0.73898613,0.74444926
,0.6979648,0.70005581,0.70658145,0.70548358,0.69179047,0.68433023
,0.70351195,0.68179868,0.70078191,0.7107512,0.72747963,0.69214428
,0.69316534,0.69065379,0.69868559,0.72469738,0.74187872,0.67538529
,0.67389448,0.6727738,0.69327232,0.67665012,0.69457393,0.69755341
,0.68413911,0.6737543,0.68560891,0.68290655,0.67940204,0.68151891
,0.68933499,0.68787329,0.6790917,0.66954938,0.67591661,0.68509305
,0.68117222,0.68695618,0.67898484,0.67710828,0.69129937,0.6728059
,0.64661095,0.66599326,0.68690894,0.67308044,0.69359962,0.66912805
,0.6554877,0.67813485,0.67933279,0.68981836,0.66002537,0.64941236
,0.66107734,0.66047074,0.65585742,0.67337856,0.67065924,0.6728124
,0.6673775,0.66043115,0.67151695,0.65177703,0.65634623,0.67514466
,0.69534124,0.65309644,0.65417753,0.66247815,0.67596904,0.65287832
,0.64811304,0.64757899,0.65710413,0.6636355,0.65966218,0.66562128
,0.64467723,0.64328003,0.65851843,0.63068687,0.64872173,0.66358774
,0.63916119,0.65041856,0.65961489,0.64523812,0.62999818,0.62299509
,0.64097496,0.63813886,0.63622686,0.65197349,0.65651046,0.65650676
,0.64458875,0.64621464,0.63036742,0.63883948,0.64634027,0.63257587
,0.62763368,0.64222574,0.63566161,0.63677233,0.64226194,0.65862405
,0.61862261,0.628845,0.62870969,0.63008999,0.63643609,0.64075678
,0.63745708,0.63905007,0.66199457,0.64009411,0.63193835,0.61948202
,0.62900756,0.63260859,0.64994605,0.66609918,0.61251223,0.60628214
,0.61356112,0.62942046,0.64179698,0.62691769,0.6103887,0.60863731
,0.62433388,0.64333705,0.64377111,0.61727128,0.62742732,0.59974787
,0.60249662,0.61598987,0.61728108,0.62698827,0.61463666,0.59842299
,0.62650485,0.60919974,0.58873,0.60798312,0.61989804,0.61897158
,0.61409212,0.60206866,0.62221292,0.61051586,0.63296702,0.63331836
,0.63283949,0.63007315,0.61595633,0.60864229,0.61235888,0.61379798
,0.58383414,0.609623,0.6274208,0.61723296,0.61359527,0.61451081
,0.61371067,0.61482123,0.61180945,0.62065176,0.60283275,0.60454569
,0.59768003,0.59612396,0.60769142,0.60225702,0.61593465,0.61875443
,0.6066403,0.61587823,0.62942211,0.59153553,0.59791366,0.61005596
,0.61793344,0.58879459,0.59270134,0.6092242,0.60380516,0.61462515
,0.62364082,0.59799179,0.59607558,0.60226395,0.58843187,0.59428786
,0.59896447,0.62166121,0.62258283,0.6211343,0.59517722,0.58980704
,0.60591672,0.60461149,0.58855613,0.57878406,0.58801699,0.58410801
,0.58643295,0.59991145,0.58988396,0.58261894,0.58708,0.5968265
,0.6049174,0.61893173,0.61244535,0.57928539,0.59713085,0.58467902
,0.59221999,0.59582941,0.59222118,0.61131593,0.58092853,0.58782018
,0.59124745,0.59279624,0.59155317,0.59632471,0.58940606,0.59702241
,0.60917609,0.58417678,0.58948877,0.59323979,0.59333992,0.59454823
,0.59819907,0.58951681,0.5994634,0.590855,0.60338276,0.58009979
,0.58837934,0.58116035,0.59727541,0.59812247,0.6049743,0.58829906
,0.60113746,0.59850564,0.57730695,0.58705884,0.59144709,0.58899187
,0.59200807,0.58456255,0.57155846,0.58236961,0.58903903,0.58161912
,0.58564733,0.58973609,0.60146552,0.6018303,0.57764548,0.58257538
,0.58044989,0.596021,0.59642148,0.61069879,0.58451381,0.59668986
,0.58454,0.60279188,0.60151976,0.6038679,0.59653159,0.57419994
,0.5759667])
gscr_arr = np.empty(0)
dscr_arr = np.empty(0)
epoch_arr =  np.arange(start=0,stop=499, step=1)

results = {'gen_loss': [], 'dis_loss': [], 'd_score': [], 'g_score': [], 'psnr': []}
for epoch in range(500, 1000):
    epoch_arr = np.append(epoch_arr, epoch)

    errG = 0.0
    errD = 0.0
    generator_loss = 0.0
    adversarial_loss = 0.0
    perceptual_loss = 0.0

    runtime_results = {'gen_loss': 0, 'dis_loss': 0, 'd_score': 0, 'g_score': 0}

    l1_loss = 0.0

    train_bar = tqdm(train_loader)

    for data, target in train_bar:
        torch.cuda.empty_cache()

        lr = data.to(DEVICE)
        hr = target.to(DEVICE)

        # lr = Variable(lr).to(DEVICE)
        # hr = Variable(hr).to(DEVICE)

        batch_size = data.size(0)

        save_image(hr, '/run/timeshift/backup/thesis/EnhanceIt/src/results/faces/real' + str(epoch) + '.png')
        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=DEVICE)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=DEVICE)

        fake_img = generator(lr).to(DEVICE)
        save_image(fake_img, '/run/timeshift/backup/thesis/EnhanceIt/src/results/faces/fake' + str(epoch) + '.png')

        discriminator.zero_grad()

        hr_output = discriminator(hr)  # Train real image.
        sr_output = discriminator(fake_img.detach())  # No train fake image.
        # Adversarial loss for real and fake images (relativistic average GAN)
        errD_hr = adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
        errD_sr = adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
        errD = errD_sr + errD_hr
        errD.backward()
        dis_score = hr_output.mean().item()
        gen_score = sr_output.mean().item()
        discriminatorOptim.step()

        ##############################################
        # (2) Update G network: E(x~real)[g(D(x))] + E(x~fake)[g(D(x))]
        ##############################################

        # Set generator gradients to zero.
        generator.zero_grad()

        # According to the feature map, the root mean square error is regarded as the content loss.
        perceptual_loss = perceptual_criterion(fake_img, hr)
        # Train with fake high resolution image.
        hr_output = discriminator(hr.detach())  # No train real fake image.
        sr_output = discriminator(fake_img)  # Train fake image.
        # Adversarial loss (relativistic average GAN)
        adversarial_loss = adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
        # Pixel level loss between two images.
        l1_loss = content_criterion(fake_img, hr)
        errG = perceptual_loss + 0.005 * adversarial_loss + 0.01 * l1_loss
        errG.backward()
        generator_loss += errG.item()/len(train_loader)
        runtime_results['gen_loss'] = generator_loss
        runtime_results['dis_loss'] = errD.item()



        D_G_z2 = sr_output.mean().item()
        optimizer.step()
        train_bar.set_description(f"[{epoch + 1}/{EPOCHS}][{epoch + 1}/{len(train_loader)}] "
                                     f"Loss_D: {errD.item():.6f} Loss_G: {generator_loss:.6f}")

    print(epoch_arr)
    print("EPOCH {} DONE: AVG. Loss: {:.4f}".format(epoch, generator_loss))
    results['dis_loss'].append(runtime_results['dis_loss'])
    results['gen_loss'].append(runtime_results['gen_loss'])

    loss_arr = np.append(loss_arr, generator_loss)
    plt.plot(epoch_arr, loss_arr, label="generator loss")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().set_xticks(epoch_arr)
    plt.savefig("esrganContinued.png")
    torch.save(generator.state_dict(),
               '/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/ESRGANsmallerblurLR0001/trainedWithFaces/generator_' + str(epoch) + ".pth")
    torch.save(discriminator.state_dict(),
               '/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/ESRGANsmallerblurLR0001/trainedWithFaces/discriminator_' + str(epoch) + ".pth")
