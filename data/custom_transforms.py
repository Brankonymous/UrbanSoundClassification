import librosa
import os, sys
import numpy as np
from speechpy import processing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import *



class ExtractMFCC1D(object):
    # Convert audio sample to MFCC Spectogram and calculate median value in every feature
    def __init__(self, num_features=NUM_MFCC_FEATURES):
        self.num_features = num_features
    
    def __call__(self, sample):
        audio_sample = sample['audio']

        # Extract MFCC
        mfcc_features = librosa.feature.mfcc(y=audio_sample, sr=sample['sample_rate'], n_mfcc=self.num_features, hop_length=5000, n_fft=2048)
        
        # Normalize and convert to 1D array
        mfcc_features = processing.cmvn(mfcc_features)
        mfcc_features = np.mean(mfcc_features, axis=1)

        sample['input'].append(mfcc_features)

        return sample


class ExtractMFCC2D(object):
    # Convert audio sample to MFCC Spectogram and calculate median value in every feature
    def __init__(self, num_features=NUM_MFCC2D_FEATURES):
        self.num_features = num_features
    
    def __call__(self, sample):
        audio_sample = sample['audio']

        # Extract MFCC
        mfcc_features = librosa.feature.mfcc(y=np.array(audio_sample), sr=sample['sample_rate'], n_mfcc=self.num_features, hop_length=4180, n_fft=2048)
        
        

        # Normalize
        mfcc_features = processing.cmvn(mfcc_features)

        
        mfcc_features = np.reshape(mfcc_features,(1,64,16))
        sample['input']=mfcc_features

        return sample


class ToTensor(object):
    # Convert sample features and output into tensor
    def __call__(self, sample):
        sample['input'] = torch.from_numpy(np.array(sample['input']).ravel())
        sample['label'] = torch.from_numpy(np.array(sample['label']))

        return sample

class ToTensorCNN(object):
    # Convert sample features and output into tensor
    def __call__(self, sample):
        sample['input'] = torch.from_numpy(np.array(sample['input']))
        sample['label'] = torch.from_numpy(np.array(sample['label']))

        return sample