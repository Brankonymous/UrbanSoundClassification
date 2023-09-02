from data.datasets import IrmasDataset
from data.custom_transforms import ExtractFFNNFeatures, ExtractMFCC, ToThreeChannels, ToTensor
from data.datasets import IrmasDataset, UrbanSounds8K
from torchvision import transforms

import numpy as np
import csv
from pandas import DataFrame

import re
import matplotlib.pyplot as plt
# from pretty_confusion_matrix import pp_matrix

import seaborn as sns


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

def loadDataset(config, val_fold=1, test=False):
    ffnn_transform = transforms.Compose([
        ExtractFFNNFeatures(),
        ToTensor()
    ])
    cnn_transform = transforms.Compose([
        ExtractMFCC(),
        ToThreeChannels(),
        ToTensor()
    ])
    if config['model_name'] == SupportedModels.FFNN.name:
        custom_transform = ffnn_transform
    elif config['model_name'] == SupportedModels.CNN.name:
        custom_transform = cnn_transform
    elif config['model_name'] == SupportedModels.VGG.name:
        custom_transform = cnn_transform

    if DATASET == 'URBAN_SOUNDS_8K':
        if not test:
            train_dataset = UrbanSounds8K(
                dataset_path = URBAN_SOUND_8K_DATASET_PATH,
                transform = custom_transform,
                train=True,
                val_fold=val_fold
            )
            val_dataset = UrbanSounds8K(
                dataset_path = URBAN_SOUND_8K_DATASET_PATH,
                transform = custom_transform,
                train=False,
                val_fold=val_fold
            )
        else:
            test_dataset = UrbanSounds8K(
                dataset_path = URBAN_SOUND_8K_DATASET_PATH,
                transform = custom_transform,
                train=True,
                val_fold=-1
            )
    
    if not test:
        return train_dataset, val_dataset
    else:
        return test_dataset

def plotImage(x, y, title='', x_label='', y_label='', flag_show=True, flag_save=True):
    # Disables showing plot if show==False
    plt.ioff()

    # Add title and axis names
    plt.figure(figsize=(15,10), facecolor='white')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot
    plt.plot(x, y)

    # Show/Save plot
    if flag_save:
        results_dir = SAVED_RESULTS_PATH + DATASET
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        
        print("Saved at:")
        print(results_dir + '/' + title + '.png')
        plt.savefig(results_dir + '/' + title + '.png')
    if flag_show:
        plt.show()

    plt.close()

def plotConfusionMatrix(confusion_matrix, val_fold, flag_show=True, title='Confusion Matrix'):
    df_cm = DataFrame(confusion_matrix, index=LABEL_NAME[:NUM_CLASSES], columns=LABEL_NAME[:NUM_CLASSES])

    # Disables showing plot if show==False
    plt.ioff()
    
    plt.figure(figsize=(15,15), facecolor='white')
    sns.heatmap(df_cm, annot=True)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    
    print("Saved at:")
    print(SAVED_RESULTS_PATH + DATASET + '/conf_matrix_' + str(val_fold) + '.png')

    results_dir = SAVED_RESULTS_PATH + DATASET
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + '/conf_matrix_' + str(val_fold) + '.png')

    if flag_show:
      plt.show()
    
    plt.close()

def plotBoxplot(data, name='', flag_show=False):
    # Disables showing plot if show==False
    plt.ioff()

    plt.figure(figsize=(15,15), facecolor='white')
    sns.boxplot(data=data, x='Accuracy')

    print("Saved at:")
    print(SAVED_RESULTS_PATH + DATASET + '/boxplot_' + name + '.png')

    results_dir = SAVED_RESULTS_PATH + DATASET
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + '/boxplot_' + name + '.png')

    if flag_show:
        plt.show()
    
    plt.close()

#Same as keras
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]