from torch import nn
from torch.utils.data import Dataset
import torchaudio
import librosa

import numpy as np
import pandas as pd
import torch
from speechpy import processing

import os, sys, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import *
from utils import utils

class IrmasDataset(Dataset):
    def __init__(self, dataset_path, transform=None, generate_csv=True):
        """
        Args:
            dataset_name (string): Path to a dataset
            transform (callable, optional): Optional audio transform to be applied
            generate_csv (bool): flag representing need for csv creation
        """
        self.dataset_csv = pd.read_csv(dataset_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_csv)
    
    def __getitem__(self, idx):
        # Get sample with index `idx`
        if (torch.is_tensor(idx)):
            idx = idx.tolist()

        path = self.dataset_csv.iloc[idx, 0]
        drums = self.dataset_csv.iloc[idx, 1]
        genre = self.dataset_csv.iloc[idx, 2]
        label = self.dataset_csv.iloc[idx, 3]

        audio_sample, sample_rate = librosa.load(path)

        # Get useful data
        sample = {
            'audio': audio_sample,
            'sample_rate': sample_rate,
            'input': [],
            'label': label
        }

        # Transform the sample
        if self.transform:
            sample = self.transform(sample)

        return sample

class UrbanSounds8K(Dataset):
    def __init__(self, dataset_path, train=True, val_fold=10, transform = None):
        self.dataset_path = dataset_path
        self.dataset_csv = pd.read_csv(self.dataset_path + URBAN_SOUND_8K_CSV_PATH)
        self.transform = transform

        self.dropSamples(train, val_fold)
    
    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, idx):
        #if (torch.is_tensor(idx)):
        #    idx = idx.tolist()

        item = self.dataset_csv.iloc[idx, 0]
        fold = self.dataset_csv.iloc[idx, 5]
        classID = self.dataset_csv.iloc[idx, 6]
        class_name = self.dataset_csv.iloc[idx,7]

        path = os.path.join(self.dataset_path, URBAN_SOUND_8K_AUDIO_PATH, 'fold' + str(fold), item)

        audio_sample, sample_rate = torchaudio.load(path)

        # TRANSFORM
        audio_sample = self.resample(audio_sample, sample_rate, SAMPLE_RATE)
        audio_sample = self.toMono(audio_sample)
        audio_sample = self.cutDown(audio_sample)
        audio_sample = self.padRight(audio_sample)

        #label = utils.to_categorical(classID,10)
        #label = torch.from_numpy(label).float()
        
        # Get useful data
        sample = {
            'audio': audio_sample,
            'sample_rate': sample_rate,
            'input': None,
            'label': classID,
            'class_name': class_name
        }

        if self.transform:
            sample = self.transform(sample)

        del sample['audio']
        del sample['sample_rate']
        del sample['class_name']

        #print(sample['label'].shape)
        return sample

    def resample(self, audio_sample, sample_rate, target_sample_rate):
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio_sample = resampler(audio_sample)
        return audio_sample

    def dropSamples(self, train, fold):
        if train:
            self.dataset_csv = self.dataset_csv[self.dataset_csv.fold != fold]

            # DEBUG #
            # self.dataset_csv = self.dataset_csv[0:100]
            # DEBUG #
        else:
            self.dataset_csv = self.dataset_csv[self.dataset_csv.fold == fold]

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



