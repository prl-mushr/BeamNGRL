import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,HEIGHT,WIDTH,CHANNELS):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 32, 3, stride=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 16, 3, stride=1, bias=False),
            nn.ELU(),
            
            nn.Conv2d(16, 8, 3, stride=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5)
        )
        x = torch.randn(CHANNELS,HEIGHT,WIDTH).view(-1,CHANNELS,HEIGHT,WIDTH)
        self._to_linear = None
        self.convs(x)
        self.dense_layers = nn.Sequential(
            nn.Linear(self._to_linear,128),
            nn.ELU(),
            nn.Dropout(p=0.5),

            nn.Linear(128,64),
            nn.ELU(),

            nn.Linear(64,32),
            nn.ELU(),

            nn.Linear(32, 2),
        )
    def convs(self, x):
        x = self.conv_layers(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.dense_layers(x)
        return x

class Bezier(nn.Module):
    def __init__(self,HEIGHT,WIDTH,CHANNELS):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 32, 3, stride=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 16, 3, stride=1, bias=False),
            nn.ELU(),
            
            nn.Conv2d(16, 8, 3, stride=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5)
        )
        x = torch.randn(CHANNELS,HEIGHT,WIDTH).view(-1,CHANNELS,HEIGHT,WIDTH)
        self._to_linear = None
        self.convs(x)
        self.dense_layers = nn.Sequential(
            nn.Linear(self._to_linear,128),
            nn.ELU(),
            nn.Dropout(p=0.5),

            nn.Linear(128,64),
            nn.ELU(),

            nn.Linear(64,32),
            nn.ELU(),

            nn.Linear(32, 5),
        )
    def convs(self, x):
        x = self.conv_layers(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.dense_layers(x)
        return x

class trajectory(nn.Module):
    def __init__(self,HEIGHT,WIDTH,CHANNELS):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.BatchNorm2d(CHANNELS),
            
            nn.Conv2d(CHANNELS, 8, 3, stride=1, bias=False),
            nn.ReLU(),
            
            nn.Conv2d(8, 16, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(16, 32, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(32, 32, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.MaxPool2d(2),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.ConvTranspose2d(32, 16, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(16, 16, 3, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, CHANNELS, 3, stride=1),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return x