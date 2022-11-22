import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LinearNeuralNetwork(nn.Module):
    def __init__(self, input_size=10):
        super(LinearNeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.ReLU(),
            nn.Linear(20, 11)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

