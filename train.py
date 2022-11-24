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
            train_dataset, val_dataset, test_dataset = utils.loadDataset(num_features=NUM_MFCC_FEATURES)

            # Generate DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            # Model
            model = LinearNeuralNetwork(input_size=NUM_MFCC_FEATURES)

            # Initialize the loss and optimizer function
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Train neural network
            for t in range(EPOCHS):
                print(f"Epoch {t+1}\n-------------------------------")
                self.trainLoop(train_dataloader, model, loss_fn, optimizer)


        elif self.config["model_name"] == SupportedModels.CNN.name:
            print("CNN")

    def trainLoop(self, dataloader, model, loss_fn, optimizer):
        # Traverse trough batches
        size = len(dataloader.dataset)
        for batch_idx, sample in enumerate(dataloader):
            X, y = sample['mfcc'], sample['label']

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
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def valLoop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for _, sample in dataloader:
                X, y = sample['mfcc'], sample['label']

                pred = model(X)
                
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
