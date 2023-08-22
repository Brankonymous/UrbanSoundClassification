import warnings
warnings.filterwarnings('ignore')

from utils import utils
from utils.constants import *
from data.custom_transforms import ExtractLinearFeatures, ExtractMFCC, ToThreeChannels, ToTensor

from torch import nn
from torch.utils.data import DataLoader, default_collate
import torchaudio
from torchvision import transforms

import numpy as np
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.linear_model import LinearNeuralNetwork


class CustomTest():
    def __init__(self,config):
        self.config = config
        self.y_true = []
        self.y_pred = []
        self.test_loss = 1
        self.accuracy = 0
        self.path = config['custom_test_path']
        self.name = config['model_name']
        self.transform_audio()

    def transform_audio(self):
        # Transform audio into universal format
        self.audio_sample, self.sample_rate = torchaudio.load(self.path)
        self.audio_sample = self.resample(self.audio_sample, self.sample_rate, SAMPLE_RATE)
        self.audio_sample = self.toMono(self.audio_sample)
        self.audio_samples = self.cutDown(self.audio_sample)
        # Transform the sample
        samples = [{
            'audio': audio_sample,
            'sample_rate': self.sample_rate,
            'input': [],
            'label': [] 
        } for audio_sample in self.audio_samples]

        sample_transform = None
        if self.config['model_name'] == SupportedModels.LINEAR.name:
            linear_transform = transforms.Compose([
                ExtractLinearFeatures(),
                ToTensor()
            ])
            sample_transform = linear_transform
        elif self.config['model_name'] == SupportedModels.CNN.name or self.config['model_name'] == SupportedModels.VGG.name:
            cnn_transform = transforms.Compose([
                ExtractMFCC(),
                ToThreeChannels(),
                ToTensor()
            ])
            sample_transform = cnn_transform

        transformed_samples = [sample_transform(sample) for sample in samples]

        # Save model input
        self.sample_input = [torch.unsqueeze(transformed_sample['input'], dim=0) for transformed_sample in transformed_samples]

    def startTest(self):
        print("Testing model")

        predicted_classes = []
        for val_fold in range(1, K_FOLD+1):
            # print("Fold: ", val_fold)

            # Model
            model_path = SAVED_MODEL_PATH + DATASET + '/' + self.config['model_name'] + "_fold" + str(val_fold) + '.pt'
            model = torch.load(model_path, map_location=torch.device(DEVICE))

            # Initialize the loss function
            loss_fn = nn.CrossEntropyLoss()

            # Start testing
            predicted_class = self.testLoop(model, loss_fn)
            predicted_classes.append(predicted_class)
        
        predicted_class = max(set(predicted_classes), key=predicted_classes.count)

        print("Classfied as: ", list(filter(lambda x: URBAN_SOUND_8K_LABEL_MAPPING[x] == predicted_class, URBAN_SOUND_8K_LABEL_MAPPING))[0])

    def testLoop(self, model, loss_fn):
        size = len(self.audio_sample)
        model.eval()

        predicted_classes = []
        with torch.no_grad():
            for ind in range(size):
                X = self.sample_input[ind].to(DEVICE)
                pred = model(X)
                
                pred = pred.argmax(1).cpu()

                predicted_classes.append(pred[0].item())

        predicted_class = max(set(predicted_classes), key=predicted_classes.count)
        return predicted_class

    def printAccuracy(self):
        print(f'Accuracy of model: {(self.accuracy/K_FOLD):>0.2f}%')

    def resample(self, audio_sample, sample_rate, target_sample_rate):
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio_sample = resampler(audio_sample)
        return audio_sample

    def toMono(self, audio_sample):
        if audio_sample.shape[0] > 1:
            audio_sample = torch.mean(audio_sample, dim=0, keepdim=True)
        return audio_sample
    
    def cutDown(self, audio_sample):
        audio_samples = []
        for ind in range(50):
            sample = audio_sample[: , ind*SAMPLE_SIZE:(ind+1)*SAMPLE_SIZE]
            
            if sample.shape[1] == SAMPLE_SIZE:
                audio_samples.append(sample)
            else:
                break
        
        return audio_samples

    def padRight(self, audio_sample):
        length = audio_sample.shape[1]

        if length < SAMPLE_SIZE:
            to_pad = SAMPLE_SIZE - length
            last_dim_padding = (0, to_pad)
            audio_sample = nn.functional.pad(audio_sample, last_dim_padding)

        return audio_sample

        

