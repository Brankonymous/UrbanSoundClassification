# InstrumentClassification

Urban Sound Classification

# Project idea
Audio classification with PyTorch. Comparing custom made FFNN and CNN models to a pre-trained VGG11 with batch normalization.

# Setup
Run <b> main.py </b> <br />
<b>NOTE</b> You should install <b>PyTorch<b> and <b>TorchAudio<b>

## Command Line Parameters
<b> --type       </b>   (Input TRAIN, TEST or CUSTOM_TEST for type of classification) <br />
<b> --model_name </b>   (Select model to use - FFNN, CNN, VGG) <br />
<b> --show_results </b>   (Plot loss and accuracy info, default=False) <br />
<b> --save_results </b>   (Plot loss and accuracy info, default=True) <br />
<b> --save_model </b>   (Save model during training, default=True) <br />
<b> --custom_test_path </b>   (Path for custom audio to classify) <br />
<br />

Info can also be found using <b>--help</b> parameter

# Network models
FFNN network with 3 hidden layers (definition can be found at `models\definitions\ffnn_model.py`) <br />
CNN network with VGG-like architecture (definition can be found at `models\definitions\cnn_model.py`)<br />
Pre-trained VGG11 with Batch Normalization <br />

# Dataset
<b> URBANSOUNDS8K </b> https://urbansounddataset.weebly.com/urbansound8k.html (browser download) <br />

<i>Taken from website: </i> <br />
<b> 10-fold cross validation using the predefined folds: train on data from 9 of the 10 predefined folds and test on data from the remaining fold </b>. Repeat this process 10 times (each time using a different set of 9 out of the 10 folds for training and the remaining fold for testing). Finally report the <b> average classification accuracy over all 10 experiments </b> (as an average score + standard deviation, or, even better, as a boxplot).
