# Multilayer Neural Blockmodeling

This repository contains the code for the paper "Neural Blockmodeling for Multilayer Networks" by Marcin Pietrasik and Marek Reformat.

## Installation

First, unpack the datasets using:

``unzip datasets.zip``

The code for our method was ran using Python version 3.6.12 along with the packages listed in ``requirements.txt``. To install these packages, run:

``pip install -r requirements.txt``

## Runtime Instructions

main.py takes three optional command line arguments:

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -h, --help 	      |	False |  shows help message and exits
| -d DATASET, --dataset DATASET          | trade          | name of dataset in datasets directory 
| -a, --A             | 8         | integer value of the A hyperparameter
| -e, --E             | 8         | integer value of the E hyperparameter
| -k, --K             | 4         | integer value of the K hyperparameter
| -t TASK, --task TASK | link_prediction | task to be performed, choosen from: link_prediction, node_classification, community_detection.
|-s SPLIT, --split SPLIT | 0.8 | float value of training split size 
| -p EPOCHS, --epochs EPOCHS | 1000 | integer value of number of epochs
|-v, --save | False | flag indicating whether to save trained model, embeddings, and community memberships

The commands for running our model using hyperparameters used in the paper are found in ``commands.txt``
