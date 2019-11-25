import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as func

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# The recurrent block is the building block of the network
class RecurrentBlock(nn.Module):

    def __init__(self, in_channel, out_channel, downsample = False, upsample = False, bottleneck = False):
        super(RecurrentBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.downsample = downsample
        self.upsample = upsample
        self.bottleneck = bottleneck

        self.hidden = None

        if self.downsample:
            self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channel, out_channel, 3, padding = 1),
                            nn.BatchNorm2d(out_channel),
                            nn.LeakyReLU(negative_slope = 0.1)
                            )
            # It takes in the output from the previous layer concatenated with the hidden layer
            self.layer2 = nn.Sequential(
                            nn.Conv2d(2*out_channel, out_channel, 3, padding = 1),
                            nn.LeakyReLU(negative_slope = 0.1),
                            nn.BatchNorm2d(out_channel),
                            nn.Conv2d(out_channel, out_channel, 3, padding = 1),
                            nn.BatchNorm2d(out_channel),
                            nn.LeakyReLU(negative_slope = 0.1)
            )

        elif self.upsample:
            self.layer1 = nn.Sequential(
					nn.Upsample(scale_factor=2, mode='nearest'),
					nn.Conv2d(2 * in_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(out_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
					nn.LeakyReLU(negative_slope=0.1),
				)

        elif self.bottleneck:
            self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channel, out_channel, 3, padding = 1),
                            nn.BatchNorm2d(out_channel),
                            nn.LeakyReLU(negative_slope=0.1)
                            )
            self.layer2 = nn.Sequential(
                            nn.Conv2d(2*out_channel, out_channel, 3, padding =1),
                            nn.BatchNorm2d(out_channel),
                            nn.LeakyReLU(negative_slope = 0.1),
                            nn.Conv2d(out_channel, out_channel, 3, padding  = 1),
                            nn.BatchNorm2d(out_channel),
                            nn.LeakyReLU(negative_slope = 0.1)
                            )

    def forward(self, X):
        if self.downsample:
             out1 = self.layer1(X)
             out2 = self.layer2(torch.cat((out1, self.hidden), dim=1))

             self.hidden = out2

             return out2

        if self.upsample:
            return self.layer1(X)

        if self.bottleneck:
            out1 = self.layer1(X)
            out2 = self.layer2(torch.cat((out1, self.hidden), dim=1))

            self.hidden = out2

            return out2

    def init_hidden(self, X, factor):
        size = list(X.size())
        size[1] = self.out_channel
        size[2] = int(size[2]/factor)
        size[3] = int(size[3]/factor)

        self.hidden_size = size
        self.hidden = torch.zeros(*(size)).to(device)


class AutoEncoder(nn.Module):

    def __init__(self, in_channel):
        super(AutoEncoder, self).__init__()

        self.encoder1 = nn.Sequential(RecurrentBlock(in_channel,32, downsample = True),
                                    nn.MaxPool2d(2))
        self.encoder2 = nn.Sequential(RecurrentBlock(32, 43, downsample = True),
                                    nn.MaxPool2d(2))
        self.encoder3 = nn.Sequential(RecurrentBlock(43, 57, downsample = True),
                                    nn.MaxPool2d(2))
        self.encoder4 = nn.Sequential(RecurrentBlock(57, 76, downsample = True),
                                    nn.MaxPool2d(2))
        self.encoder5 = nn.Sequential(RecurrentBlock(76, 101, downsample = True),
                                    nn.MaxPool2d(2))

        self.bottleneck = RecurrentBlock(101, 101, bottleneck = True)

        self.decoder5 = RecurrentBlock(101, 76, upsample = True)
        self.decoder4 = RecurrentBlock(76, 57, upsample = True)
        self.decoder3 = RecurrentBlock(57, 43, upsample = True)
        self.decoder2 = RecurrentBlock(43, 32, upsample = True)
        self.decoder1 = RecurrentBlock(32, 3, upsample = True)

        self.pool = nn.MaxPool2d(kernel_size = 2)

    def set_input(self, X):
        self.input = X

    def forward(self):
        encoder1 = self.encoder1(self.input)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        bottleneck = self.bottleneck(encoder5)

        decoder5 = self.decoder5(torch.cat((bottleneck, encoder5), dim=1))
        decoder4 = self.decoder4(torch.cat((decoder5, encoder4), dim=1))
        decoder3 = self.decoder3(torch.cat((decoder4, encoder3), dim=1))
        decoder2 = self.decoder2(torch.cat((decoder3, encoder2), dim=1))
        decoder1 = self.decoder1(torch.cat((decoder2, encoder1), dim=1))

        return decoder1

    def reset_hidden(self):

        self.encoder1[0].init_hidden(self.input, 1)
        self.encoder2[0].init_hidden(self.input, 2)
        self.encoder3[0].init_hidden(self.input, 4)
        self.encoder4[0].init_hidden(self.input, 8)
        self.encoder5[0].init_hidden(self.input, 16)

        self.bottleneck.init_hidden(self.input, 32)
