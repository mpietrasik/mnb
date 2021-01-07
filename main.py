#Mute tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser
from model import *
from numpy import argmax
from numpy import array
from numpy import load
from numpy import mean
from numpy import std
from numpy import squeeze
from numpy import unique
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utils import *

def link_prediction(model, inputs_testing, outputs_testing):

    #Predict interaction of testing data and calculate ROC AUC
    results = model.predict(generate_input(inputs_testing))
    print('Link Prediction ROC AUC Score :', roc_auc_score(squeeze(outputs_testing), squeeze(results)), sep='\t')

def node_classification(embedding_model, communities):

    #Generate embeddings for testing data
    embeddings = []
    for node in range(communities.shape[0]):
        embedding = embedding_model.predict(array([node]))
        embeddings.append(squeeze(embedding))
    
    #Classify nodes by their embeddings and print the results
    scores = classify(array(embeddings), communities, 0.8)
    print('Node Classification accuracy average:', mean(scores), sep='\t')
    print('Node Classification accuracy standard deviation', std(scores), sep='\t')

def community_detection(community_membership_model, edges, K):
    
    #Generate community memberships for testing data
    community_memberships = []
    for node in range(unique(edges[:,0]).shape[0]):
        community_membership = community_membership_model.predict(array([node]))
        community_memberships.append(argmax(community_membership))
    
    #Calculate normalized cut and conductance and print the results
    normalized_cut = calculate_normalized_cut(edges, community_memberships, K)
    print('Community Detection normalized cut :', normalized_cut)
    conductance = calculate_conductance(edges, community_memberships, K)
    print('Community Detection conductance :', conductance)

if __name__ == '__main__':

    #Interpret command line arguments
    argument_parser = ArgumentParser(description='Generate subsumption axioms for document-tag pairs')
    argument_parser.add_argument('-d', '--dataset', help='Name of dataset in datasets directory (default = trade)', default = 'trade', type=str)
    argument_parser.add_argument('-a', '-A',  help='Integer value of the A hyperparameter (default = 8)', default=8, type=int)
    argument_parser.add_argument('-e', '-E', help='Integer value of the E hyperparameter (default = 8)', default=8, type=int)
    argument_parser.add_argument('-k', '-K', help='Integer value of the K hyperparameter (default = 4)', default=4, type=int)
    argument_parser.add_argument('-t', '--task', help='Task to be performed. Choose from: link_prediction, node_classification, community_detection. (default = link_prediction)', default='link_prediction', choices=['link_prediction', 'node_classification', 'community_detection'])
    argument_parser.add_argument('-s', '--split', help='Float value of training split size (default = 0.8)', default=0.8, type=float)
    argument_parser.add_argument('-p', '--epochs', help='Int value of number of epochs (default = 1000)', default=1000, type=int)    
    argument_parser.add_argument('-v', '--save', help='Flag indicating whether to save trained model, embeddings, and community memberships', action='store_true')
    arguments = argument_parser.parse_args()

    dataset = arguments.dataset
    A = arguments.a
    E = arguments.e
    K = arguments.k
    task = arguments.task
    split = arguments.split
    epochs = arguments.epochs
    save = arguments.save

    #Handle command line parameters
    if split <= 0.0 or 1.0 < split:
        raise ValueError('Training split size must be in range (0,1]')

    if task == 'link_prediction' and (split <= 0.0 or 1.0 <= split):
        raise ValueError('Training split size must be in range (0,1) for link prediciton task')

    if epochs < 1:
        raise ValueError('Number of epochs must be greater than or equal to 1')

    print('--------MULTILAYER NETWORK EMBEDDING--------')
    print('Dataset :', dataset, sep='\t')
    print('A :', A, sep='\t')
    print('E :', E, sep='\t')
    print('K :', K, sep='\t')
    print('Task :', task, sep='\t')
    print('Training split size :', split, sep='\t')
    print('Number of epochs :', epochs, sep='\t')
    print()

    for iteration in range(5): #remove later

        #Load datasets
        edges = load('datasets/' + dataset + '_edges.npy')
        communities = load('datasets/' + dataset + '_communities.npy')

        #Calculate number of nodes (N) and number fo relations (R) from dataset
        N = unique(edges[:,0]).shape[0]
        R = relations = unique(edges[:,1]).shape[0]

        #Preprocess dataset
        inputs, outputs = preprocess_data(edges)

        #Create training and testing splits
        if split == 1:
            inputs_training, inputs_testing, outputs_training, outputs_testing = inputs, None, outputs, None
        else:    
            inputs_training, inputs_testing, outputs_training, outputs_testing = train_test_split(inputs, outputs, test_size = 1 - split)

        #Initialize model
        model, embedding_model, community_membership_model = initialize_model(A, E, K, N, R)

        #Train model
        print('Training model (this may take some time)')
        model.fit(generate_input(inputs_training), squeeze(outputs_training), epochs = epochs, verbose = 0, shuffle=True)
        print('Training complete')
        print()

        #Evaluate trained model for the specified task
        if task == 'link_prediction':
            link_prediction(model, inputs_testing, outputs_testing)
        elif task == 'node_classification':
            node_classification(embedding_model, communities)
        elif task == 'community_detection':
            community_detection(community_membership_model, edges, K)

        #If indicated, save trained model, embeddings, and community memberships
        if save:
            model.save('MNB-Trained-Model_Dataset-' + dataset + '_A-' + str(A) + '_E-' + str(E) + '_K-' + str(K))
            save_embeddings(embedding_model, N, dataset, E)
            save_community_memberships(community_membership_model, N, dataset, K)
        
