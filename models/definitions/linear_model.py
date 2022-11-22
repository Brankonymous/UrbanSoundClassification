import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LinearNeuralNetwork():
    def __init__(self, input_size=10):
        super(LinearNeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLu(),
            nn.Linear(16, 20),
            nn.ReLu(),
            nn.Linear(20, 16)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

