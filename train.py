from utils.constants import *

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        if self.config["model_type"] == SupportedModels.LINEAR.name:
            print("LINEAR")
        
        

        elif self.config["model_type"] == SupportedModels.CNN.name:
            print("CNN")
        