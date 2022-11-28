import librosa
import os, sys
import numpy as np
from speechpy import processing
from utils.constants import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import *

class ExtractLinearFeatures(object):
    # Convert audio sample to MFCC Spectogram and calculate median value in every feature
    def __init__(self, num_features=NUM_MFCC_FEATURES):
        self.num_features = num_features
    
    def __call__(self, sample):
        audio_sample = sample['audio']
        sample_rate = sample['sample_rate']

        if FLAG_RMS:
            rms = np.mean(librosa.feature.rms(y=audio_sample))
            sample['input'].append(rms)
        if FLAG_SPEC_CENT:
            spec_cen = np.mean(librosa.feature.spectral_centroid(y=audio_sample, sr=sample_rate))
            sample['input'].append(spec_cen)
        if FLAG_SPEC_BW:
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio_sample, sr=sample_rate))
            sample['input'].append(spec_bw)
        if FLAG_ROLLOF:
            rollof = np.mean(librosa.feature.spectral_rolloff(y=audio_sample, sr=sample_rate))
            sample['input'].append(rollof)
        if FLAG_ZERO_CR:
            zero_cr = np.mean(librosa.feature.zero_crossing_rate(y=audio_sample))
            sample['input'].append(zero_cr)
        if NUM_MFCC_FEATURES:
            # Extract MFCC
            mfcc_features = librosa.feature.mfcc(y=audio_sample, sr=sample_rate, n_mfcc=self.num_features)
            
            # Normalize and convert to 1D array
            mfcc_features = processing.cmvn(mfcc_features)
            mfcc_features = np.mean(mfcc_features, axis=1)

            [sample['input'].append(feature) for feature in mfcc_features]

        sample['input'] = np.array(sample['input']).ravel()
        return sample


class ExtractMFCC(object):
    # Convert audio sample to MFCC Spectogram and calculate median value in every feature
    def __init__(self, num_features=NUM_MFCC_FEATURES):
        self.num_features = num_features
    
    def __call__(self, sample):
        audio_sample = sample['audio']
        #print(audio_sample.shape,sample['label'])
        # Extract MFCC
        mfcc_features = librosa.feature.mfcc(y=np.array(audio_sample), sr=sample['sample_rate'], n_mfcc=self.num_features, hop_length = 1024)
        # Normalize

        mfcc_features = processing.cmvn(mfcc_features)
       
        #print(mfcc_features.shape)
        mfcc_features = np.reshape(mfcc_features, (1, mfcc_features.shape[0], mfcc_features.shape[1]))
        #print(mfcc_features.shape)
        sample['input'] = mfcc_features

        return sample

class ToTensor(object):
    # Convert sample features and output into tensor
    def __call__(self, sample):
        sample['input'] = torch.from_numpy(np.array(sample['input']))
        sample['label'] = torch.from_numpy(np.array(sample['label']))

        return sample