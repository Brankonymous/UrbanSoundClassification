import warnings
warnings.filterwarnings('ignore')

from utils import utils
from utils.constants import *

from torch import nn
from torch.utils.data import DataLoader, default_collate

import numpy as np
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.ffnn_model import FFNNNeuralNetwork

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.loss = []
        self.accuracy, self.recall, self.precision, self.F1 = [], [], [], []

    def startTrain(self, val_fold):
        # torch.multiprocessing.set_start_method('spawn')
        print("Training model \n")
            
        # Initialize dataset
        train_dataset, val_dataset = utils.loadDataset(config=self.config, val_fold=val_fold)

        # Generate DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) #shuffle = True / changed to false
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) #shuffle = True / changed to false

        if self.config['model_name'] == SupportedModels.FFNN.name:
            # Model
            input_size = NUM_MFCC_FEATURES + FLAG_RMS + FLAG_ROLLOF + FLAG_SPEC_CENT + FLAG_SPEC_BW + FLAG_ZERO_CR
            model = FFNNNeuralNetwork(input_size=input_size)

            # Initialize the loss and optimizer function
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.1)

            ###### END OF Fully Connected Neural Network #######
                
        elif self.config['model_name'] == SupportedModels.CNN.name:
            # Model
            model = ConvNeuralNetwork()
            
            # Initialize the loss and optimizer function
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.4)

            ###### END OF CNN #######

        elif self.config['model_name'] == SupportedModels.VGG.name:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=False)
            # Initialize the loss and optimizer function
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.1)

            ###### END OF VGG #######

        # Train and validate neural network
        model.to(DEVICE)
        start_time = datetime.datetime.now()
        for t in range(EPOCHS):
            # Print info
            curr_time = datetime.datetime.now()
            print(f'Epoch {t+1}\n-------------------------------')
            print(f'Current time: {curr_time.time()} / Time elapsed from beggining: {curr_time-start_time}')

            # Train and validate epoch
            self.trainLoop(train_dataloader, model, loss_fn, optimizer, scheduler)
            self.valLoop(val_dataloader, model, loss_fn)

            # Check if model is learning compared to previous epoch
            # if len(self.loss) != 0 and self.loss[-1] > epoch_loss:
            #    break

        # Plot and save results if flags are true
        if self.config['show_results'] or self.config['save_results']:
            flag_show = True if self.config['show_results'] else False
            flag_save = True if self.config['save_results'] else False

            title_info = '(' + self.config['model_name'] + ' model)' + ' Val. fold: ' + str(val_fold)
            utils.plotImage(x=np.arange(len(self.loss), dtype=np.int64), y=self.loss, title='Loss ' + title_info, x_label = 'Epochs', y_label='Cross entropy loss', flag_show=flag_show, flag_save=flag_save)
            utils.plotImage(x=np.arange(len(self.accuracy), dtype=np.int64), y=self.accuracy, title='Accuracy ' + title_info, x_label = 'Epochs', y_label='Accuracy (%)', flag_show=flag_show, flag_save=flag_save)
            utils.plotImage(x=np.arange(len(self.precision), dtype=np.int64), y=self.precision, title='Precision ' + title_info, x_label = 'Epochs', y_label='Accuracy (%)', flag_show=flag_show, flag_save=flag_save)
            utils.plotImage(x=np.arange(len(self.recall), dtype=np.int64), y=self.recall, title='Recall ' + title_info, x_label = 'Epochs', y_label='Accuracy (%)', flag_show=flag_show, flag_save=flag_save)
            utils.plotImage(x=np.arange(len(self.F1), dtype=np.int64), y=self.F1, title='F1 Score ' + title_info, x_label = 'Epochs', y_label='Accuracy (%)', flag_show=flag_show, flag_save=flag_save)

        # Save model if flag is true
        if self.config['save_model'] or self.config['type'] == ModelType.TRAIN_AND_TEST.name:
            torch.save(model, SAVED_MODEL_PATH + DATASET + '/' + self.config['model_name'] + '_fold' + str(val_fold) + '.pt')

    def trainLoop(self, dataloader, model, loss_fn, optimizer, scheduler):
        # Traverse trough batches
        size = len(dataloader.dataset)
        model.train()
        accuracy = 0
        print(next(model.parameters()).device)

        for batch_idx, sample in enumerate(dataloader):
            X, y = sample['input'], sample['label']
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred.argmax(1)
            accuracy += (pred == y).sum().item()

            # Output results
            if batch_idx % 10 == 0:
                loss, current = loss.item(), min(size, batch_idx * BATCH_SIZE)
                print(f'loss: {loss:>7f}  [{current}/{size}], lr: {scheduler.get_last_lr()}')
                # print(pred)
                print('--------------------------------------------------')
        accuracy *= 100/size
        print(f'Train accuracy: {accuracy:>7f}')

        scheduler.step()

    def valLoop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        model.eval()
        
        num_batches = len(dataloader)
        y_pred, y_true = [], []
        val_loss, accuracy, precision, recall, F1 = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                X, y = sample['input'], sample['label']

                pred = model(X)
                
                val_loss += loss_fn(pred, y).item()
                pred = pred.argmax(1)
                
                accuracy += (pred == y).sum().item()

                y_pred.append(pred.cpu().numpy())
                y_true.append(y.cpu().numpy())
        
        val_loss /= num_batches
        accuracy /= size

        y_pred = np.concatenate(np.array(y_pred))
        y_true = np.concatenate(np.array(y_true))
        
        recall = recall_score(y_true = y_true, y_pred=y_pred, average='weighted', labels=np.unique(y_pred))
        precision = precision_score(y_true = y_true, y_pred=y_pred, average='weighted', labels=np.unique(y_pred))
        F1 = f1_score(y_true = y_true, y_pred=y_pred, average='weighted', labels=np.unique(y_pred))

        accuracy *= 100
        recall *= 100
        precision *= 100
        F1 *= 100

        print(f'Validation Error: \n Accuracy: {(accuracy):>0.1f}%, Avg Loss: {val_loss:>8f} \n')
        print(f' Recall: {(recall):>0.1f}%, Precision: {(precision):>0.1f}%, F1 Score: {(F1):>0.1f}% \n')

        # Append results
        self.loss.append(val_loss)
        self.accuracy.append(accuracy)
        self.recall.append(recall)
        self.precision.append(precision)
        self.F1.append(F1)
