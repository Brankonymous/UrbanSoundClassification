from torch import nn
from utils.constants import *

class LinearNeuralNetwork(nn.Module):
    def __init__(self, input_size=10):
        super(LinearNeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        
        self.linearReluStack = nn.Sequential(
            nn.Linear(input_size, 25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.ReLU(),
            nn.Linear(20, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # ???
        x = x.to(torch.float32)
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

