import librosa
import os, sys
import numpy as np
import torchaudio
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
        audio_sample = sample['audio'].numpy()
        sample_rate = sample['sample_rate']
        sample['input'] = []
        print(audio_sample.shape)

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

            # Normalize
            # mfcc_features = processing.cmvn(mfcc_features)
            if len(mfcc_features.shape) > 2:
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

        ### IRMAS ###
        # Extract MFCC
        #mfcc_features = librosa.feature.mfcc(y=np.array(audio_sample), sr=sample['sample_rate'], n_mfcc=self.num_features, hop_length = 1024)
        #mfcc_features = processing.cmvn(mfcc_features)
        #mfcc_features = np.reshape(mfcc_features, (1, mfcc_features.shape[0], mfcc_features.shape[1]))
        ### IRMAS ###

        transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=self.num_features,
            melkwargs={"n_fft": 1024, "hop_length": 512},
        )
        mfcc_features = transform(audio_sample)

        sample['input'] = mfcc_features

        return sample

class ToThreeChannels(object):
    def __call__(self, sample):
        audio = sample['input']

        y_dim = audio.shape[1]
        z_dim = audio.shape[2]

        audio = audio.expand(3, y_dim, z_dim)

        sample['input'] = audio
        return sample

class ToTensor(object):
    # Convert sample features and output into tensor
    def __call__(self, sample):
        sample['input'] = torch.from_numpy(np.array(sample['input']))
        sample['label'] = torch.from_numpy(np.array(sample['label']))

        return sample