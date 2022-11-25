import librosa
import os, sys
import numpy as np
from speechpy import processing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import *

class ExtractMFCC(object):
    # Convert audio sample to MFCC Spectogram and calculate median value in every feature
    def __init__(self, num_features=NUM_MFCC_FEATURES):
        self.num_features = num_features
    
    def __call__(self, sample):
        audio_sample = sample['audio']

        mfcc_features = librosa.feature.mfcc(y=audio_sample, sr=sample['sample_rate'], n_mfcc=self.num_features)
        mfcc_features = processing.cmvn(mfcc_features)
        mfcc_features = np.median(mfcc_features, axis=1)

        # Normalize feature
        mfcc_features = mfcc_features / np.linalg.norm(mfcc_features)
        # print(mfcc_features)

        sample['mfcc'] = mfcc_features

        return sample


class ToTensor(object):
    # Covert sample features and output into tensor

    def __call__(self, sample):
        sample['mfcc'] = torch.from_numpy(sample['mfcc'])
        sample['label'] = torch.from_numpy(np.array(sample['label']))

        return sample