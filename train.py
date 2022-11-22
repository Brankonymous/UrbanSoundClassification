from utils import utils
from utils.constants import *

from torch import nn
from torch.utils.data import DataLoader

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.linear_model import LinearNeuralNetwork

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        if self.config["model_name"] == SupportedModels.LINEAR.name:
            # Initialize dataset
            irmas_dataset = utils.loadDataset(num_features=NUM_MFCC_FEATURES)

            # Generate DataLoader
            dataloader = DataLoader(irmas_dataset, batch_size=4, shuffle=True, num_workers=0)

            # Model
            model = LinearNeuralNetwork(input_size=LINEAR_STARTING_NODE_SIZE)

            X = torch.rand(11, 1, 10, device=DEVICE)
            logits = model(X)
            pred_probab = nn.Softmax(dim=1)(logits)
            y_pred = pred_probab.argmax(1)
            print(f"Predicted class: {y_pred}")

            # Traverse trough batches
            for i_batch, sample_batched in enumerate(dataloader):
                print(i_batch, sample_batched['mfcc'].size(), sample_batched['label'].size())


        elif self.config["model_name"] == SupportedModels.CNN.name:
            print("CNN")

    
        