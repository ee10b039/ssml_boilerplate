#!usr/bin/env python
"""
The architecture docstring
"""

import os
import torch

__author__ = "Vinay Kumar"
__copyright__ = "copyright 2018, Project SSML"
__maintainer__ = "Vinay Kumar"
__status__ = "Research & Development"

# SANITY-CHECK: if the wav folder exists
#######################################

# initial Input tensor size = (batch, channel = 1, height = 15, width = 1025)
# final Output tensor size = (batch, channel = 1, height = 15, width = 1025)

class Interpolate(torch.nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interpolate = torch.nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(input=x, size=self.size, mode=self.mode, align_corners=False)
        return x

class Save_layer(torch.nn.Module):
    def __init__(self):
        super(Save_layer, self).__init__()


    def forward(self, x):
        # print or save the layer
        # print(x)
        return x

class MeeEncoder(torch.nn.Module):
    def __init__(self):
        super(MeeEncoder, self).__init__()

        self.en_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3,3), stride=(1,1), bias=True)
        self.en_l1_bn  = torch.nn.BatchNorm2d(num_features=12)
        self.en_layer2 = torch.nn.MaxPool2d(kernel_size=(5,3),stride=(1,1))
        self.en_layer3 = torch.nn.Conv2d(in_channels=12, out_channels=20, kernel_size=(3,3), stride=(1,1), bias=True)
        self.en_l3_bn  = torch.nn.BatchNorm2d(num_features=20)
        self.en_layer4 = torch.nn.MaxPool2d(kernel_size=(5,1), stride=(1,1))
        self.en_layer5 = torch.nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3,3), stride=(1,1), bias=True)
        self.en_l5_bn  = torch.nn.BatchNorm2d(num_features=30)
        self.en_layer6 = torch.nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3,3), stride=(1,1), bias=True)
        self.en_l6_bn  = torch.nn.BatchNorm2d(num_features=40)
        self.en_ReLU = torch.nn.ReLU()
        self.en_leakyReLU = torch.nn.LeakyReLU(negative_slope=1e-02)
        self.save_layer = Save_layer()

        self.encoder = torch.nn.Sequential(
            self.en_layer1,
            self.save_layer,
            self.en_l1_bn,
            self.save_layer,
            self.en_leakyReLU,
            self.save_layer,
            self.en_layer2,
            self.save_layer,
            self.en_layer3,
            self.save_layer,
            self.en_l3_bn,
            self.save_layer,
            self.en_leakyReLU,
            self.save_layer,
            self.en_layer4,
            self.save_layer,
            self.en_layer5,
            self.save_layer,
            self.en_l5_bn,
            self.save_layer,
            self.en_leakyReLU,
            self.save_layer,
            self.en_layer6,
            self.save_layer,
            self.en_l6_bn,
            self.save_layer,
            self.en_leakyReLU,
            self.save_layer
        )

    def forward(self, x):
        return self.encoder(x)

class MeeDecoder(torch.nn.Module):
    def __init__(self):
        super(MeeDecoder, self).__init__()

        self.de_layer1 = torch.nn.Conv2d(in_channels=40, out_channels=30, kernel_size=(3,3), stride=(1,1), bias=True)
        self.de_l1_bn  = torch.nn.BatchNorm2d(num_features=30)
        self.de_layer2 = torch.nn.Conv2d(in_channels=30, out_channels=20, kernel_size=(3,3), stride=(1,1), bias=True)
        self.de_l2_bn  = torch.nn.BatchNorm2d(num_features=20)
        self.de_layer3 = Interpolate(size=(205, 5), mode='bilinear')
        self.de_layer4 = torch.nn.Conv2d(in_channels=20, out_channels=12, kernel_size=(3,3), stride=(1,1), bias=True)
        self.de_l4_bn  = torch.nn.BatchNorm2d(num_features=12)
        self.de_layer5 = Interpolate(size=(1025, 15), mode='bilinear')
        self.de_layer6 = torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)
        # self.de_l5_bn  = torch.nn.BatchNorm2d(num_features=1)
        self.de_ReLU = torch.nn.ReLU()
        self.de_leakyReLU = torch.nn.LeakyReLU(negative_slope=1e-02)
        self.save_layer = Save_layer()

        self.decoder = torch.nn.Sequential(
            self.de_layer1,
            self.save_layer,
            self.de_l1_bn,
            self.save_layer,
            self.de_leakyReLU,
            self.save_layer,
            self.de_layer2,
            self.save_layer,
            self.de_l2_bn,
            self.save_layer,
            self.de_leakyReLU,
            self.save_layer,
            self.de_layer3,
            self.save_layer,
            self.de_layer4,
            self.save_layer,
            self.de_l4_bn,
            self.save_layer,
            self.de_leakyReLU,
            self.save_layer,
            self.de_layer5,
            self.save_layer,
            self.de_layer6,
            self.save_layer,
            self.de_leakyReLU,
            self.save_layer
        )

    def forward(self, x):
        return self.decoder(x)

class MeeAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(MeeAutoEncoder, self).__init__()
        self.encoder = MeeEncoder()
        self.decoder = MeeDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

#########--------------------------------------------------------------------
#########--------------------------------------------------------------------


