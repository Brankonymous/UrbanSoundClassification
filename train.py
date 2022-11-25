from utils import utils
from utils.constants import *

from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.linear_model import LinearNeuralNetwork

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.loss = []
        self.accuracy = []

    def startTrain(self):
        if self.config['model_name'] == SupportedModels.LINEAR.name:
            # Initialize dataset
            train_dataset, val_dataset, test_dataset = utils.loadDataset(n_mfcc=NUM_MFCC_FEATURES)

            # Generate DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

            # Model
            model = LinearNeuralNetwork(input_size=NUM_MFCC_FEATURES)

            # Initialize the loss and optimizer function
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Train and validate neural network
            for t in range(EPOCHS):
                print(f'Epoch {t+1}\n-------------------------------')
                self.trainLoop(train_dataloader, model, loss_fn, optimizer)
                epoch_loss, epoch_accuracy = self.valLoop(val_dataloader, model, loss_fn)

                self.loss.append(epoch_loss)
                self.accuracy.append(epoch_accuracy)

            if self.config['show_results'] or self.config['save_results']:
                show_flag = True if self.config['show_results'] else False
                save_flag = True if self.config['save_results'] else False

                utils.plotImage(x=np.arange(EPOCHS, dtype=np.int64), y=self.loss, title='Loss (' + self.config['model_name'] + ' model)', x_label = 'Epochs', y_label='Cross entropy loss', show=show_flag, save=save_flag)
                utils.plotImage(x=np.arange(EPOCHS, dtype=np.int64), y=self.accuracy, title='Accuracy (' + self.config['model_name'] + ' model)', x_label = 'Epochs', y_label='Accuracy (%)', show=show_flag, save=save_flag)

            if self.config['save_model'] or self.config['type'] == ModelType.TRAIN_AND_TEST.name:
                torch.save(model, SAVED_MODEL_PATH + self.config['model_name'] + '.pt')

        elif self.config['model_name'] == SupportedModels.CNN.name:
            pass

    def trainLoop(self, dataloader, model, loss_fn, optimizer):
        # Traverse trough batches
        size = len(dataloader.dataset)
        for batch_idx, sample in enumerate(dataloader):
            X, y = sample['input'], sample['label']

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Output results
            if batch_idx % 10 == 0:
                loss, current = loss.item(), batch_idx * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    def valLoop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, accuracy = 0, 0

        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                X, y = sample['input'], sample['label']

                pred = model(X)
                
                test_loss += loss_fn(pred, y).item()
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        accuracy /= size
        accuracy *= 100 * 100
        print(f'Validation Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n')

        return test_loss, accuracy
