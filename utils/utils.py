from data.preprocess_data import IrmasDataset
from data.custom_transforms import ExtractMFCC, ToTensor
from torchvision import transforms

import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import matplotlib


from sklearn.model_selection import train_test_split

from .constants import *

def writeToCSV(csv_path, header, data):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rows in data:
            writer.writerow(rows)

def parseIrmasDataset(irmas_csv_path, dataset_path):
    # Create annotated csv
    header = ['path', 'drums', 'genre', 'label']
    X, y = [], []
    # data = ["/", "nod", "cla", "pia"]
    DEBUG_CSV_SIZE = 100
         
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
                    X.append([path, drums, genre])
                    y.append([label])

                    if len(X) == DEBUG_CSV_SIZE:
                        break
                if len(X) == DEBUG_CSV_SIZE:
                        break
            if len(X) == DEBUG_CSV_SIZE:
                        break
        if len(X) == DEBUG_CSV_SIZE:
                        break

    # Converting arrays to numpy
    X = np.array(X)
    y = np.array(y)

    # Train/Validation/Test split (80/10/10)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, stratify=y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, stratify=y_rem, test_size=0.5)

    # Write data to csv
    writeToCSV(csv_path=irmas_csv_path+'irmas_train.csv', header=header, data=np.append(X_train, y_train, axis=1))
    writeToCSV(csv_path=irmas_csv_path+'irmas_val.csv', header=header, data=np.append(X_val, y_val, axis=1))
    writeToCSV(csv_path=irmas_csv_path+'irmas_test.csv', header=header, data=np.append(X_test, y_test, axis=1))

def loadDataset(n_mfcc):
    train_dataset = IrmasDataset(
        dataset_path=IRMAS_DATASET_DIRECTORY + 'irmas_train.csv',
        transform = transforms.Compose([
            ExtractMFCC(num_features=n_mfcc),
            ToTensor()
        ]),
        generate_csv=True
    )
    val_dataset = IrmasDataset(
        dataset_path=IRMAS_DATASET_DIRECTORY + 'irmas_val.csv',
        transform = transforms.Compose([
            ExtractMFCC(num_features=n_mfcc),
            ToTensor()
        ]),
        generate_csv=True
    )
    test_dataset = IrmasDataset(
        dataset_path=IRMAS_DATASET_DIRECTORY + 'irmas_test.csv',
        transform = transforms.Compose([
            ExtractMFCC(num_features=n_mfcc),
            ToTensor()
        ]),
        generate_csv=True
    )

    return train_dataset, val_dataset, test_dataset

def plotImage(x, y, title='', x_label='', y_label='', show=True, save=False):
    # Disables showing plot if show==False
    plt.ioff()

    # Add title and axis names
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot
    plt.plot(x, y)

    # Show/Save plot
    if save:
        plt.savefig(SAVED_RESULTS_PATH + title + '.png')
    if show:
        plt.show()

    plt.close()