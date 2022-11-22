
from data.preprocess_data import IrmasDataset
from data.custom_transforms import ExtractMFCC, ToTensor
from torchvision import transforms

from .constants import *

def loadDataset(num_features):
    irmas_dataset = IrmasDataset(
        root_dir=IRMAS_DATASET_DIRECTORY, 
        dataset_path=IRMAS_SINGLE_INST_DATASET_PATH,
        transform = transforms.Compose([
            ExtractMFCC(num_features=num_features),
            ToTensor()
        ]),
        generate_csv=True
    )
    return irmas_dataset
