import os
import argparse
import shutil
import time

import numpy as np
import torch

from train import TrainNeuralNetwork
from test import TestNeuralNetwork

import utils.utils as utils
from utils.constants import *

def train(config):
    for val_fold in range(1, K_FOLD+1):
        print(f'--------- Validation fold {val_fold} ---------')
        
        trainNeuralNet = TrainNeuralNetwork(config=config)
        trainNeuralNet.startTrain(val_fold)

def test(config):
    testNeuralNet = TestNeuralNetwork(config=config)
    for val_fold in range(1, K_FOLD+1):
      testNeuralNet.startTest(val_fold, flag_show=config['show_results'])

def custom_test(config):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common params
    # Izbrisali smo argparse.BooleanOptionalAction
    parser.add_argument('--type', choices=[m.name for m in ModelType], type=str, help='Input TRAIN, TEST or CUSTOM_TEST for type of classification', default=ModelType.TRAIN_AND_TEST.name)
    parser.add_argument('--model_name', choices=[m.name for m in SupportedModels], type=str, help='Neural network (model) to use', default=SupportedModels.VGG.name) #default=SupportedModels.LINEAR.name
    parser.add_argument('--show_results', help='Plot loss and accuracy info during training', default=False)
    parser.add_argument('--save_results', help='Save loss and accuracy info during training', default=True)
    parser.add_argument('--save_model', help='Save model during training', default=True)

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
    

    