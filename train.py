from utils import utils
from utils.constants import *

from torch.utils.data import DataLoader

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        if self.config["model_name"] == SupportedModels.LINEAR.name:
            # Initialize dataset
            irmas_dataset = utils.loadDataset(num_features=NUM_MFCC_FEATURES)

            # Generate DataLoader
            dataloader = DataLoader(irmas_dataset, batch_size=4, shuffle=True, num_workers=0)

            # Traverse trough batches
            for i_batch, sample_batched in enumerate(dataloader):
                print(i_batch, sample_batched['mfcc'].size(), sample_batched['label'].size())

                

        elif self.config["model_name"] == SupportedModels.CNN.name:
            print("CNN")

    
        