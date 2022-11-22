from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
from torchvision import transforms
from .custom_transforms import ExtractMFCC, ToTensor
import librosa

import numpy as np
import pandas as pd
import csv

import os, sys, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import *

class IrmasDataset(Dataset):
    def __init__(self, root_dir, dataset_path, transform=None, generate_csv=True):
        """
        Args:
            root_dir (string): Directory of all IRMAS datasets
            dataset_path (string): Path to a dataset
            transform (callable, optional): Optional audio transform to be applied
            generate_csv (bool): flag representing need for csv creation
        """
        self.parse_irmas_dataset(root_dir, dataset_path, generate_csv)
        self.dataset_csv = pd.read_csv(root_dir + "/irmas_dataset.csv")

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_csv)
    
    def __getitem__(self, idx):
        # Get sample with index `idx`

        if (torch.is_tensor(idx)):
            idx = idx.tolist()

        path = self.dataset_csv.iloc[idx, 0]
        drums = self.dataset_csv.iloc[idx, 1]
        genre = self.dataset_csv.iloc[idx, 2]
        label = self.dataset_csv.iloc[idx, 3]

        audio_sample, sample_rate = librosa.load(self.dataset_csv.iloc[idx, 0])

        # Get useful data
        sample = {
            'path': path, 
            'audio': audio_sample,
            'sample_rate': sample_rate,
            'drums': drums,
            'genre': genre,
            'label': label
        }

        # Transform the sample
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def parse_irmas_dataset(self, root_dir, dataset_path, generate_csv):
        # Create annotated csv
        if generate_csv:
            header = ['path', 'drums', 'genre', 'label']
            # data = ["/", "nod", "cla", "pia"]

            with open(root_dir + "/irmas_dataset.csv", 'w') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(header)
                
                # Write data
                for root, subdirectories, _ in os.walk(dataset_path):
                    for subdirectory in subdirectories:
                        for subdir_root, _, files in os.walk(os.path.join(root, subdirectory)):
                            for file in files:
                                # Default file params
                                path = os.path.join(subdir_root, file)
                                drums = "nod"
                                genre = None
                                label = LABEL_MAPPING[subdirectory]

                                # Extract category info from file name
                                categories = re.findall('\[.*?\]', file)[1:]
                                categories = [category.replace('[', '').replace(']', '') for category in categories]

                                for category in categories:
                                    if category == "nod" or category == "dru":
                                        drums = category
                                    else:
                                        genre = category
                                    
                                # Write to csv
                                data = [path, drums, genre, label]
                                writer.writerow(data)



