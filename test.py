from utils import utils
from utils.constants import *

from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.linear_model import LinearNeuralNetwork

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.confusion_matrix = np.array(NUM_CLASSES, NUM_CLASSES)

    def startTest(self):
        print("Testing model \n")
        if self.config['model_name'] == SupportedModels.LINEAR.name:
            # Initialize dataset
            _, _, test_dataset = utils.loadDataset(n_mfcc=NUM_MFCC_FEATURES, config=self.config)

            # Generate DataLoader
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

            # Model
            model = torch.load(SAVED_MODEL_PATH + self.config['model_name'] + '.pt')

            # Initialize the loss function
            loss_fn = nn.CrossEntropyLoss()

            self.testLoop(test_dataloader, model, loss_fn)
        
        elif self.config['model_name'] == SupportedModels.CNN.name:
                pass

    def testLoop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        model.eval()

        num_batches = len(dataloader)
        test_loss, accuracy = 0, 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                X, y = sample['input'], sample['label']

                pred = model(X)
                
                test_loss += loss_fn(pred, y).item()
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Debug output
                print(f'Testing...: [{min(size, batch_idx * BATCH_SIZE)}/{size}]')


        test_loss /= num_batches
        accuracy /= size
        accuracy *= 100
        print(f'Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n')

        return test_loss, accuracy
