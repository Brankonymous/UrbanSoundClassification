import enum
import torch
import sys
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_DIRECTORY = 'data/dataset/'
DATASET = 'URBAN_SOUNDS_8K'

SAVED_MODEL_PATH = 'models/saved_models/'
SAVED_RESULTS_PATH = 'data/results/'

# Common
NUM_CLASSES = 10

# Model constants
NUM_WORKERS = 0
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
LR_STEP_SIZE = 5
WEIGHT_DECAY = 0

# Fully Connected Network features
FLAG_RMS = True
FLAG_SPEC_CENT = True
FLAG_SPEC_BW = True
FLAG_ROLLOF = True
FLAG_ZERO_CR = True

# Fully Connected Network and CNN Feature
NUM_MFCC_FEATURES = 50 # Changed from 50 to 52

# DATASET SPECIFIC

# IRMAS DATASET
IRMAS_DATASET_PATH = 'data/dataset/IRMAS-TrainingData/'
IRMAS_LABEL_NAME = ['flute', 'trumpet', 'organ', 'piano', 'electric guitar', 'saxophone', 'voice', 'clarinet', 'acoustic guitar', 'cello', 'violin']
IRMAS_LABEL_MAPPING = {
    'flu': 0,
    'tru': 1,
    'gel': 2,
    'org': 3,
    'pia': 4,
    'sax': 5,
    'voi': 6,
    'cla': 7,
    'gac': 8,
    'cel': 9,
    'vio': 10
}

# We take 2 seconds of each audio file
SAMPLE_RATE = 44100
NUM_SAMPLES = 2 * SAMPLE_RATE # 1s

# UrbanSound8K DATASET
NUM_CLASSES_8K = 10
K_FOLD = 2
URBAN_SOUND_8K_DATASET_PATH = 'data/dataset/UrbanSound8K/'
URBAN_SOUND_8K_CSV_PATH = 'metadata/UrbanSound8K.csv'
URBAN_SOUND_8K_AUDIO_PATH = 'audio/'

URBAN_SOUND_8K_LABEL_NAME = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
URBAN_SOUND_8K_LABEL_MAPPING = {
    'air_conditioner': 0,
    'car_horn': 1,
    'children_playing': 2,
    'dog_bark': 3,
    'drilling': 4,
    'engine_idling': 5,
    'gun_shot': 6,
    'jackhammer': 7,
    'siren': 8,
    'street_music': 9
}

####
PHILHARM_DATASET_PATH = 'data/dataset/Philharmonia'
PHILHARM_LABEL_NAME = ['double bass', 'flute', 'guitar', 'saxophone', 'trumpet', 'violin']
PHILHARM_LABEL_MAPPING = {
    'double bass': 0,
    'flute': 1,
    'guitar': 2,
    'saxophone': 3,
    'trumpet': 4,
    'violin': 5
}

if DATASET == 'IRMAS':
    DATASET_PATH = IRMAS_DATASET_PATH
    LABEL_NAME = IRMAS_LABEL_NAME
    LABEL_MAPPING = IRMAS_LABEL_MAPPING
elif DATASET == 'PHILHARMONIA':
    DATASET_PATH = PHILHARM_DATASET_PATH
    LABEL_NAME = PHILHARM_LABEL_NAME
    LABEL_MAPPING = PHILHARM_LABEL_MAPPING
elif DATASET == 'URBAN_SOUNDS_8K':
    DATASET_PATH = URBAN_SOUND_8K_DATASET_PATH
    LABEL_NAME = URBAN_SOUND_8K_LABEL_NAME
    LABEL_MAPPING = URBAN_SOUND_8K_LABEL_MAPPING
    NUM_CLASSES = NUM_CLASSES_8K


class ModelType(enum.Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_AND_TEST = 2
    CUSTOM_TEST = 3

class SupportedModels(enum.Enum):
    LINEAR = 0
    CNN = 1
    VGG = 2
