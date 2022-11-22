import enum
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelType(enum.Enum):
    TRAIN = 0
    TEST = 1
    CUSTOM_TEST = 2

class SupportedModels(enum.Enum):
    LINEAR = 0
    CNN = 1

SUPPORTED_AUDIO_FORMATS = ["wav", "mp3"]

# TODO Check the directories