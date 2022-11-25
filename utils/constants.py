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

# Model constants
NUM_WORKERS = 0
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-1
LR_STEP_SIZE = 10
WEIGHT_DECAY = 1e-5

<<<<<<< HEAD
# CNN constants
IMAGE_SIZE = 128


NUM_MFCC_FEATURES = 20
=======
NUM_MFCC_FEATURES = 30
>>>>>>> a949362b3d3a7c70e9b41dd1cbac01db93b75d57


LABEL = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
LABEL_MAPPING = {
    "cel": 0,
    "cla": 1,
    "flu": 2,
    "gac": 3,
    "gel": 4,
    "org": 5,
    "pia": 6,
    "sax": 7,
    "tru": 8,
    "vio": 9,
    "voi": 10
}
REAL_LABEL_MAPPING = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic guitar",
    "gel": "electric guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "voice"
}

class ModelType(enum.Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_AND_TEST = 2
    CUSTOM_TEST = 3

class SupportedModels(enum.Enum):
    LINEAR = 0
    CNN = 1

SUPPORTED_AUDIO_FORMATS = ["wav", "mp3"]