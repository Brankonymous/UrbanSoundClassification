import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.constants import *

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding = 'same'
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding = 'same'
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(
                in_channels=1024,
                out_channels=2048,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.adaptivePool = nn.AdaptiveMaxPool2d((4, 4))

        self.linearStack = nn.Sequential(
            nn.Linear(16384, 64),
            # nn.Linear(1024 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        #x = self.adaptivePool(x)
        x = self.flatten(x)
        x = self.linearStack(x)
        
        
        predictions = self.softmax(x)
        
        return predictions