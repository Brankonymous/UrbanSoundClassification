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
from sklearn.metrics import f1_score, precision_score, recall_score

from models.definitions.cnn_model import ConvNeuralNetwork
from models.definitions.linear_model import LinearNeuralNetwork


class CustomTest():
    def __init__(self,config):
        self.path = config['custom_test_path']
        self.name = config['model_name']
        self.audio_sample, self.sample_rate = torchaudio.load(self.path)
        self.audio_sample = self.resample(self.audio_sample, self.sample_rate, SAMPLE_RATE)
        self.audio_sample = self.toMono(self.audio_sample)
        self.audio_sample = self.cutDown(self.audio_sample)
        self.audio_sample = self.padRight(self.audio_sample)
        sample = {
            'audio': self.audio_sample,
            'sample_rate': self.sample_rate,
            'input': [],
            'label': []      # Mozda pravi problem
        }
        cnn_transform = transforms.Compose([
            ExtractMFCC(),
            ToThreeChannels(),
            ToTensor()
        ])

        X = cnn_transform(sample)
        self.audio_sample = np.array(X['input'])
        self.audio_sample = torch.from_numpy(np.expand_dims(self.audio_sample, axis=0))
        # self.audio_sample = np.array([np.newaxis, ...])
        print(self.audio_sample.shape)

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
        if audio_sample.shape[1] > NUM_SAMPLES:
            audio_sample = audio_sample[: ,:NUM_SAMPLES]
        return audio_sample

    def padRight(self, audio_sample):
        length = audio_sample.shape[1]

        if length < NUM_SAMPLES:
            to_pad = NUM_SAMPLES - length
            last_dim_padding = (0, to_pad)
            audio_sample = nn.functional.pad(audio_sample, last_dim_padding)

        return audio_sample
    
    def get_name(self,ans):

        if ans == 0:
            return 'air_conditioner'
        if ans == 1:
            return 'car_horn'
        if ans == 2:
            return 'children_playing'
        if ans == 3:
            return 'air_conditioner'
        if ans == 4:
            return 'dog_bark'
        if ans == 5:
            return 'engine_idling'
        if ans == 6:
            return 'gun_shot'
        if ans == 7:
            return 'jackhammer'
        if ans == 8:
            return 'siren'
        if ans == 9:
            return 'street_music'

    def custom_test(self):
        answer = np.zeros(NUM_CLASSES)
        for i in range(1, K_FOLD+1):
            model_path = SAVED_MODEL_PATH + DATASET + '/' + self.name + "_fold" + str(i) + '.pt'
            # print(model_path)
            model = torch.load(model_path, map_location=torch.device(DEVICE))

            pred = model(self.audio_sample)
            pred = pred[0][:10].detach()

            answer = np.add(answer, np.array(pred))
        print('Model predicted: ', self.get_name(answer.argmax()))

        

