import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.constants import *

# TODO add CNN architecture
# obrisan komentar


#Random for now
#IMAGE_SIZE = 128

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(64 * IMAGE_SIZE/4 * IMAGE_SIZE/4),
            nn.Linear(64 * IMAGE_SIZE/4 * IMAGE_SIZE/4,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 11),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.model(x)
