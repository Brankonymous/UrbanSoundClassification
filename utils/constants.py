import enum
import torch
import sys
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IRMAS_DATASET_DIRECTORY = "data/dataset/"
IRMAS_SINGLE_INST_DATASET_PATH = "data/dataset/IRMAS-TrainingData/"
IRMAS_MULTI_INST_DATASET_PATH = "data/dataset/IRMAS-TestingData-Part1/"

SAVED_MODEL_PATH = "models/saved_models/"
SAVED_RESULTS_PATH = "data/results/"

# Common
NUM_CLASSES = 2

# Model constants
NUM_WORKERS = 0
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-5
LR_STEP_SIZE = 5
WEIGHT_DECAY = 0

# Fully Connected Network features
FLAG_RMS = True
FLAG_SPEC_CENT = True
FLAG_SPEC_BW = True
FLAG_ROLLOF = True
FLAG_ZERO_CR = True

# Fully Connected Network and CNN Feature
NUM_MFCC_FEATURES = 26

LABEL_NAME = ["flute", "trumpet", "organ", "piano", "electric guitar", "saxophone", "voice", "clarinet", "acoustic guitar", "cello", "violin"]
LABEL_MAPPING = {
    "flu": 0,
    "tru": 1,
    "gel": 2,
    "org": 3,
    "pia": 4,
    "sax": 5,
    "voi": 6,
    "cla": 7,
    "gac": 8,
    "cel": 9,
    "vio": 10
}

class ModelType(enum.Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_AND_TEST = 2
    CUSTOM_TEST = 3

class SupportedModels(enum.Enum):
    LINEAR = 0
    CNN = 1
