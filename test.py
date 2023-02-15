from utils import utils
from utils.constants import *

from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.linear_model import LinearNeuralNetwork
from sklearn.metrics import confusion_matrix

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.y_true = []
        self.y_pred = []
        self.test_loss = 1/K_FOLD
        self.accuracy = 1/K_FOLD

    def startTest(self, val_fold, flag_show=True):
        print("Testing model " + str(val_fold))
            
        # Initialize dataset
        test_dataset = utils.loadDataset(config=self.config, test=True)

        # Generate DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Model
        model_path = SAVED_MODEL_PATH + DATASET + '/' + self.config['model_name'] + "_fold" + str(val_fold) + '.pt'
        model = torch.load(model_path)

        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()

        test_loss, accuracy = self.testLoop(test_dataloader, model, loss_fn)
        utils.plotConfusionMatrix(self.conf_mat, val_fold, flag_show)

        self.test_loss *= test_loss
        self.accuracy *= accuracy


    def testLoop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        model.eval()

        num_batches = len(dataloader)
        test_loss, accuracy, precision, recall, F1 = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                X, y = sample['input'], sample['label']
                X, y = X.to(DEVICE), y.to(DEVICE)

                pred = model(X)
                
                test_loss += loss_fn(pred, y).item()
                
                pred = pred.argmax(1)
                pred, y = pred.cpu(), y.cpu()
                accuracy += (pred == y).type(torch.float).sum().item()
                recall += recall_score(y_true = y.numpy(), y_pred=pred.numpy(), average='micro', labels=np.unique(pred))
                precision += precision_score(y_true = y.numpy(), y_pred=pred.numpy(), average='micro', labels=np.unique(pred))
                F1 += f1_score(y_true = y.numpy(), y_pred=pred.numpy(), average='micro', labels=np.unique(pred))

                # Debug output
                if batch_idx % 10 == 0:
                  print(f'Testing...: [{min(size, batch_idx * BATCH_SIZE)}/{size}]')

                # Save results
                [self.y_pred.append(val) for val in pred.numpy()]
                [self.y_true.append(val) for val in y.numpy()]

        
        self.conf_mat = confusion_matrix(self.y_true, self.y_pred, labels=np.arange(NUM_CLASSES))
        
        test_loss /= num_batches
        accuracy /= size

        accuracy *= 100


        print(f'Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg Loss: {test_loss:>8f} \n')
        print(f' Recall: {(recall):>0.1f}%, Precision: {(precision):>0.1f}%, F1 Score: {(F1):>0.1f}% \n')

        return test_loss, accuracy
