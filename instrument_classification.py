import os
import argparse
import shutil
import time

import numpy as np
import torch

import utils.utils as utils
from utils.constants import *

def train(config):
    pass

def test(config):
    pass

def custom_test(config):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument("--type", choices=[m.name for m in ModelType], type=str, help="Input TRAIN, TEST or CUSTOM_TEST for type of classification", default=ModelType.TRAIN.name)
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels], type=str, help="Neural network (model) to use", default=SupportedModels.LINEAR.name)

    # Wrapping configuration into a dictionary
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    
    if (config['type'] == "TRAIN"):
        train(config)
    elif (config['type'] == "TEST"):
        test(config)
    elif (config['type'] == "CUSTOM_TEST"):
        custom_test(config)
    