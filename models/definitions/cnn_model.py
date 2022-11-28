import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.constants import *

class ConvNeuralNetwork(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.convStack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.4),
            nn.Flatten()
        )
        self.linearStack = nn.Sequential(
            nn.Linear(64*13*11, 10),
            torch.nn.ReLU(),
            nn.Linear(10,20),
             nn.ReLU(),
            torch.nn.Linear(20, NUM_CLASSES_8K),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = self.convStack(x)
        x = self.linearStack(x)
        
        #print(x)
        return x