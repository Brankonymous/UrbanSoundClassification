from torch import nn
from utils.constants import *

class FFNNNeuralNetwork(nn.Module):
    def __init__(self, input_size=10):
        super(FFNNNeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        
        self.ffnnReluStack = nn.Sequential(
            nn.Linear(input_size, 25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.ReLU(),
            nn.Linear(20, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.flatten(x)
        logits = self.ffnnReluStack(x)
        return logits

