from torch import nn
from torch.utils.data import Dataset
import torchaudio
import librosa

import numpy as np
import pandas as pd

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
    def __init__(self, dataset_items_path, dataset_csv_path ,num_samples, sample_rate, transform = None):

        self.dataset_csv = pd.read_csv(dataset_csv_path)
        self.transform = transform
        self.dataset_items_path = dataset_items_path
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.target_sample_rate = sample_rate
    
    def __len__(self):
        return len(self.dataset_csv)

    # ( 1 , x ) --> (1 , self.num_samples )
    def _cut_down_if_needed(self,signal):
        if signal.shape[0] > self.num_samples:
            signal = signal[:self.num_samples]
        return signal

    def _pad_right_if_needed(self,signal):
        length = signal.shape[0]

        if length < self.num_samples:
            to_pad = self.num_samples - length
            last_dim_padding = (0, to_pad)

            signal = nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_needed(self,signal,sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def __getitem__(self, idx):

        if (torch.is_tensor(idx)):
            idx = idx.tolist()

        item = self.dataset_csv.iloc[idx, 0]
        fold = self.dataset_csv.iloc[idx, 5]
        classID = self.dataset_csv.iloc[idx, 6]
        class_name = self.dataset_csv.iloc[idx,7]

        path = os.path.join(self.dataset_items_path, 'fold' + str(fold), item)

        audio_sample, sample_rate = librosa.load(path)

        #PADDING TRANSFORMS 

        audio_sample = self._resample_if_needed(audio_sample,sample_rate)
        audio_sample = torch.from_numpy(audio_sample)
        audio_sample = self._cut_down_if_needed(audio_sample)
        audio_sample = self._pad_right_if_needed(audio_sample)

        label = utils.to_categorical(classID,10)
        label = torch.from_numpy(label)

        # Get useful data
        sample = {
            'audio': audio_sample,
            'sample_rate': sample_rate,
            'input': [],
            'label': label,
            'class_name':class_name
        }
        #print(sample['label'].shape)

        if self.transform:
            sample = self.transform(sample)

        return sample
