from torch import nn
import torch, torchvision
import math

# ----------------------------------------
# --------------ESRGAN MODEL---------------
# ----------------------------------------

class ESRGAN(nn.Module):
    def __init__(self,scale = 2):
        super(ESRGAN, self).__init__()
        self.in_channels = 3
        self.nf = 64
        self.growth_rate = 32
        self.block_num = 23
        self.out_channels = 3

        self.block1 = nn.Sequential(nn.ReflectionPad2d(padding=1), nn.Conv2d(self.in_channels, self.nf, kernel_size=3,
                                                                             stride=1, padding=0), nn.ReLU(inplace=True))

        block_layers = []
        for _ in range(self.block_num):
            block_layers += [RIDB(self.nf, self.growth_rate)]

        self.block = nn.Sequential(*block_layers)
        self.block2 = nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True))
        self.upsample = Upscale_Block(self.nf, scale)
        self.block3 = nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True))
        self.block4 = nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(self.nf, self.out_channels, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.block1(x)
        out = self.block(x1)
        out = self.block2(out)
        out = self.upsample(out + x1)
        out = self.block3(out)
        out = self.block4(out)
        return out



class Discriminator_ESRGAN(nn.Module):
    def __init__(self, num_block = 4):
        super(Discriminator_ESRGAN, self).__init__()
        block = []
        in_channels = 3
        out_channels = 64


# Residual Dense Block
class RDB(nn.Module):
    def __init__(self,in_channels, growth_rate = 32):
        super(RDB,self).__init__()

        #number of input channels
        #  growth rate is the number of filters to add each layer

        self.b1 = nn.Sequential(nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2, inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2, inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels +  2 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2, inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(in_channels +  3 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2, inplace=True))
        self.b5 = nn.Sequential(nn.Conv2d(in_channels +  4 * growth_rate, in_channels, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True))

    def forward(self,x ):

        b1 = self.b1(x)
        b2 = self.b2(torch.cat((x,b1), dim=1))
        b3 = self.b3(torch.cat((x,b1,b2),dim=1))
        b4 = self.b4 (torch.cat((x,b1,b2,b3),dim=1))
        b5 = self.b5(torch.cat((x,b1,b2,b3,b4),dim=1))
        return  b5.mul(0.2) + x

#ResidualInResidualDenseBlock
class RIDB(nn.Module):
    def __init__(self, nf, growth_rate = 32):
        super(RIDB, self).__init__()
        self.l1 = RDB(nf,growth_rate)
        self.l2 = RDB(nf,growth_rate)
        self.l3 = RDB(nf, growth_rate)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out.mul(0.2) + x


def Upscale_Block(nf, scale = 2):
    block = []
    for _ in range(scale//2):
        block+= [
            nn.Conv2d(nf,nf * (2**2), 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]
    return nn.Sequential(*block)



# ----------------------------------------
# --------------SRGAN MODEL---------------
# ----------------------------------------


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, 0.8)
        )
        block8 = [UpsamplingBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Dropout2d(0.3)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


# ----------------------------------------
# --------------SRCNN MODEL---------------
# ----------------------------------------

class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


# ----------------------------------------
# --------------ARCNN MODEL---------------
# ------Used for artifact reduction-------
# ----------------------------------------


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x
