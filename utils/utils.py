from data.datasets import IrmasDataset
from data.custom_transforms import ExtractLinearFeatures, ToTensor, mel_spectrogram
from data.datasets import IrmasDataset, UrbanSounds8K
from torchvision import transforms

import numpy as np
import csv
from pandas import DataFrame

import re
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix

import seaborn as sn

from sklearn.model_selection import train_test_split

from .constants import *

def writeToCSV(csv_path, header, data):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rows in data:
            writer.writerow(rows)

def parseDataset(csv_path, dataset_path):
    # Create annotated csv
    # data = ['/', 'nod', 'cla', 'pia']
    header = ['path', 'drums', 'genre', 'label']
    X, y = [], []
    label_num = np.zeros(NUM_CLASSES)

    # Write data
    for root, subdirectories, _ in os.walk(dataset_path):
        for subdirectory in subdirectories:
            for subdir_root, _, files in os.walk(os.path.join(root, subdirectory)):
                for file in files:
                    # Default file params
                    path = os.path.join(subdir_root, file)
                    drums = ''
                    genre = None
                    label = LABEL_MAPPING[subdirectory]

                    if label >= NUM_CLASSES:
                        continue

                    # Extract category info from file name
                    categories = re.findall('\[.*?\]', file)[1:]
                    categories = [category.replace('[', '').replace(']', '') for category in categories]

                    for category in categories:
                        if category == 'nod' or category == 'dru':
                            drums = category
                        else:
                            genre = category
                    
                    if drums == 'dru':
                        continue

                    # Write to csv
                    X.append([path, drums, genre])
                    y.append([label])
                    label_num[label] += 1

                    if len(X) >= 100000:
                        break
                if len(X) >= 100000:
                    break
            if len(X) >= 100000:
                break
        if len(X) >= 100000:
            break

    print(LABEL_NAME[:NUM_CLASSES])
    print(label_num)
    min_class = label_num.min()

    # Rebalance dataset
    label_num = np.zeros(NUM_CLASSES)
    bal_X, bal_y = [], []
    for i in range(len(X)):
        if label_num[y[i][0]] < min_class:
            label_num[y[i][0]] += 1
            bal_X.append(X[i])
            bal_y.append(y[i])

    print("Dataset balanced ")
    print(label_num, label_num.sum())

    # Converting arrays to numpy
    X = np.array(bal_X)
    y = np.array(bal_y)

    # Train/Validation/Test split (80/10/10)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, stratify=y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, stratify=y_rem, test_size=0.5)

    # Write data to csv
    writeToCSV(csv_path=csv_path+'irmas_train.csv', header=header, data=np.append(X_train, y_train, axis=1))
    writeToCSV(csv_path=csv_path+'irmas_val.csv', header=header, data=np.append(X_val, y_val, axis=1))
    writeToCSV(csv_path=csv_path+'irmas_test.csv', header=header, data=np.append(X_test, y_test, axis=1))

def loadDataset(config):
    if config['model_name'] == SupportedModels.LINEAR.name:
        train_dataset = IrmasDataset(
            dataset_path=DATASET_DIRECTORY + 'irmas_train.csv',
            transform = transforms.Compose([
                ExtractLinearFeatures(),
                ToTensor()
            ]),
            generate_csv=True
        )
        val_dataset = IrmasDataset(
            dataset_path=DATASET_DIRECTORY + 'irmas_val.csv',
            transform = transforms.Compose([
                ExtractLinearFeatures(),
                ToTensor()
            ]),
            generate_csv=True
        )
        test_dataset = IrmasDataset(
            dataset_path=DATASET_DIRECTORY + 'irmas_test.csv',
            transform = transforms.Compose([
                ExtractLinearFeatures(),
                ToTensor()
            ]),
            generate_csv=True
        )
        
    #To be changed if it works

    elif config['model_name'] == SupportedModels.CNN.name:
        train_dataset = UrbanSounds8K(
            dataset_items_path = URBAN_SOUND_8K_PATH_AUDIO,
            dataset_csv_path = URBAN_SOUND_8K_PATH_META_bigger,
            num_samples = NUM_SAMPLES,
            sample_rate = SAMPLE_RATE,
            transform = mel_spectrogram
            # generate_csv=True
        )
        val_dataset = UrbanSounds8K(
            dataset_items_path = URBAN_SOUND_8K_PATH_AUDIO,
            dataset_csv_path = URBAN_SOUND_8K_PATH_META_bigger,
            num_samples = NUM_SAMPLES,
            sample_rate = SAMPLE_RATE,
            transform = mel_spectrogram
            # generate_csv=True
        )
        test_dataset = UrbanSounds8K(
            dataset_items_path = URBAN_SOUND_8K_PATH_AUDIO,
            dataset_csv_path = URBAN_SOUND_8K_PATH_META_bigger,
            num_samples = NUM_SAMPLES,
            sample_rate = SAMPLE_RATE,
            transform = mel_spectrogram
            # generate_csv=True
        )
        #     train_dataset = IrmasDataset(
        #     dataset_path=DATASET_DIRECTORY + 'irmas_train.csv',
        #     transform = transforms.Compose([
        #         ExtractMFCC(),
        #         ToTensor()
        #     ]),
        #     generate_csv=True
        # )
        #     val_dataset = IrmasDataset(
        #     dataset_path=DATASET_DIRECTORY + 'irmas_val.csv',
        #     transform = transforms.Compose([
        #         ExtractMFCC(),
        #         ToTensor()
        #     ]),
        #     generate_csv=True
        # )
        #     test_dataset = IrmasDataset(
        #     dataset_path=DATASET_DIRECTORY + 'irmas_test.csv',
        #     transform = transforms.Compose([
        #         ExtractMFCC(),
        #         ToTensor()
        #     ]),
        #     generate_csv=True
        # )

    return train_dataset, val_dataset, test_dataset

def plotImage(x, y, title='', x_label='', y_label='', flag_show=True, flag_save=True):
    # Disables showing plot if show==False
    plt.ioff()

    # Add title and axis names
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot
    plt.plot(x, y)

    # Show/Save plot
    if flag_save:
        plt.savefig(SAVED_RESULTS_PATH + title + '.png')
    if flag_show:
        plt.show()

    plt.close()

def plotConfusionMatrix(confusion_matrix, title='Confusion Matrix'):
    df_cm = DataFrame(confusion_matrix, index=LABEL_NAME[:NUM_CLASSES], columns=LABEL_NAME[:NUM_CLASSES])

    pp_matrix(df_cm, cmap='Oranges', annot=True)
    plt.close()

#Same as keras
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]