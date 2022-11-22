import librosa
import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import *

class ExtractMFCC(object):
    def __init__(self, num_features=10):
        self.num_features = num_features
    
    def __call__(self, sample):
        audio_sample = sample['audio']

        mfcc_features = librosa.feature.mfcc(y=audio_sample, sr=sample['sample_rate'], n_mfcc=self.num_features)
        mfcc_features = np.median(mfcc_features, axis=1)

        sample['mfcc'] = mfcc_features

        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['mfcc'] = torch.from_numpy(sample['mfcc'])
        sample['label'] = torch.from_numpy(np.array(sample['label']))

        return sample