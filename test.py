from utils import utils
from utils.constants import *

from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import seaborn as sns

from sklearn.metrics import confusion_matrix

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.y_true = []
        self.y_pred = []
        self.test_loss = 1/K_FOLD
        self.accuracy = []

    def startTest(self, val_fold, flag_show=True):
        print("Testing model " + str(val_fold))
            
        # Initialize dataset
        _, test_dataset = utils.loadDataset(config=self.config, val_fold=val_fold)

        # Generate DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load model
        model_path = SAVED_MODEL_PATH + DATASET + '/' + self.config['model_name'] + "_fold" + str(val_fold) + '.pt'
        model = torch.load(model_path, map_location=torch.device(DEVICE))

        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()

        test_loss, accuracy = self.testLoop(test_dataloader, model, loss_fn)
        utils.plotConfusionMatrix(self.conf_mat, val_fold, flag_show)

        self.test_loss *= test_loss
        self.accuracy.append(accuracy)


    def testLoop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        model.eval()

        num_batches = len(dataloader)
        test_loss, accuracy, precision, recall, F1 = 0, 0, 0, 0, 0

        y_pred, y_true = [], []
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                X, y = sample['input'], sample['label']
                X, y = X.to(DEVICE), y.to(DEVICE)

                pred = model(X)
                
                test_loss += loss_fn(pred, y).item()
                
                pred = pred.argmax(1)
                pred, y = pred.cpu(), y.cpu()
                accuracy += (pred == y).type(torch.float).sum().item()
                # recall += recall_score(y_true = y.numpy(), y_pred=pred.numpy(), average='micro', labels=np.unique(pred))
                # precision += precision_score(y_true = y.numpy(), y_pred=pred.numpy(), average='micro', labels=np.unique(pred))
                # F1 += f1_score(y_true = y.numpy(), y_pred=pred.numpy(), average='micro', labels=np.unique(pred))

                # Debug output
                if batch_idx % 10 == 0:
                  print(f'Testing...: [{min(size, batch_idx * BATCH_SIZE)}/{size}]')

                # Save results
                [self.y_pred.append(val) for val in pred.numpy()]
                [self.y_true.append(val) for val in y.numpy()]

                y_pred.append(pred.cpu().numpy())
                y_true.append(y.cpu().numpy())

        
        self.conf_mat = confusion_matrix(self.y_true, self.y_pred, labels=np.arange(NUM_CLASSES))
        
        test_loss /= num_batches
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

        print(f'Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg Loss: {test_loss:>8f} \n')
        print(f' Recall: {(recall):>0.1f}%, Precision: {(precision):>0.1f}%, F1 Score: {(F1):>0.1f}% \n')

        return test_loss, accuracy

    def printAccuracy(self):
        self.accuracy = np.array(self.accuracy)
        print(f'Accuracy of model: {(self.self.accuracy.mean()):>0.2f}%')
        print(f'Standard deviation: {(self.self.accuracy.std()):>0.2f}')
        utils.plotBoxplot(self.accuracy, name=self.config['model_name'], flag_show=self.config['show_results'])