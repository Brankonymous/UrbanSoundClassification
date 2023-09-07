from data.custom_transforms import ExtractFFNNFeatures, ExtractMFCC, ToThreeChannels, ToTensor, NormalizeFFNN, NormalizeCNN
from data.datasets import UrbanSounds8K
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


def loadDataset(config, val_fold=1, test=False):
    ffnn_transform = transforms.Compose([
        ExtractFFNNFeatures(),
        ToTensor(),
    ])
    cnn_transform = transforms.Compose([
        ExtractMFCC(),
        ToThreeChannels(),
        ToTensor(),
    ])
    vgg_transform = transforms.Compose([
        ExtractMFCC(),
        ToThreeChannels(),
        ToTensor(),
    ])
    
    if config['model_name'] == SupportedModels.FFNN.name:
        custom_transform = ffnn_transform
    elif config['model_name'] == SupportedModels.CNN.name:
        custom_transform = cnn_transform
    elif config['model_name'] == SupportedModels.VGG.name:
        custom_transform = vgg_transform

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
    
    sns.boxplot(data = data)

    print("Saved at:")
    print(SAVED_RESULTS_PATH + DATASET + '/boxplot_' + name + '.png')

    results_dir = SAVED_RESULTS_PATH + DATASET
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + '/boxplot_' + name + '.png')

    if flag_show:
        plt.show()
    
    plt.close()

def plotBarplot(data, name = '', flag_show = False):
    plt.ioff()

    plt.figure(figsize=(15,15), facecolor='white')
    data = list(data)
    plt.bar(range(1,11), data)
    #print('GAS', data, type(data))

    print("Saved at:")
    print(SAVED_RESULTS_PATH + DATASET + '/barplot_' + name + '.png')

    results_dir = SAVED_RESULTS_PATH + DATASET
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + '/barplot_' + name + '.png')

    if flag_show:
        plt.show()
    
    plt.close()



#Same as keras
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]