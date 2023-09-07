import os
import argparse
import shutil
import time

import numpy as np
import torch

from train import TrainNeuralNetwork
from test import TestNeuralNetwork
from test_custom import CustomTest

import utils.utils as utils
from utils.constants import *
import datetime

def train(config):
    start = datetime.datetime.now()
    for val_fold in range(1, K_FOLD+1):
        print('Time from REAL beginig: ', datetime.datetime.now() - start)
        print(f'--------- Validation fold {val_fold} ---------')
        
        trainNeuralNet = TrainNeuralNetwork(config=config)
        trainNeuralNet.startTrain(val_fold)
    print('Training lasted for: ', datetime.datetime.now() - start)

def test(config):
    testNeuralNet = TestNeuralNetwork(config=config)
    for val_fold in range(1, K_FOLD+1):
      testNeuralNet.startTest(val_fold, flag_show=config['show_results'])

    testNeuralNet.printAccuracy()

def custom_test(config):
    # For test purposes - data\custom_audio\dog_barking.mp3

    customTest = CustomTest(config)

    customTest.startTest()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common params
    # Izbrisali smo argparse.BooleanOptionalAction
    parser.add_argument('--type', choices=[m.name for m in ModelType], type=str, help='Input TRAIN, TEST or CUSTOM_TEST for type of classification', default=ModelType.TRAIN_AND_TEST.name)
    parser.add_argument('--model_name', choices=[m.name for m in SupportedModels], type=str, help='Neural network (model) to use', default=SupportedModels.CNN.name)
    parser.add_argument('--show_results', help='Plot loss and accuracy info', default=False)
    parser.add_argument('--save_results', help='Save loss and accuracy info', default=False)
    parser.add_argument('--save_model', help='Save model during training', default=True)
    parser.add_argument('--custom_test_path', help= 'Path for custom audio to classify', default='')
    
    # Wrapping configuration into a dictionary
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    
    if config['type'] == 'TRAIN' or config['type'] == 'TRAIN_AND_TEST':
        train(config)
    if config['type'] == 'TEST' or config['type'] == 'TRAIN_AND_TEST':
        test(config)
    if config['type'] == 'CUSTOM_TEST':
        custom_test(config)
    

    