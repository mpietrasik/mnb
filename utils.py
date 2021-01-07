from networkx import conductance
from networkx import DiGraph
from networkx import normalized_cut_size
from numpy import array
from numpy import min as npmin
from numpy import random
from numpy import save
from numpy import squeeze
from numpy import unique
from numpy import zeros
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def preprocess_data(edges):

    #Separate inputs from outputs
    inputs = edges[:, 0:3]
    outputs = edges[:, 3]
    return inputs, outputs

def generate_input(input_data):

    #Format input so it can be fed into the network
    return [input_data[:,0],input_data[:,1], input_data[:,2], zeros((input_data.shape[0],1))]

def classify(embeddings, communities, split):

    #sanity check
    assert embeddings.shape[0] == communities.shape[0]

    #Run linear SVM classifier 10 times and save accuracy scores
    scores = []
    for iteration in range(10):

        #Split nodes into training and testing
        training_input, testing_input, training_output, testing_output = train_test_split(embeddings, communities, test_size = 1 - split)

        #Run linear SVM classifier and calculate accuracy
        try:
            classifier = svm.SVC(decision_function_shape='ovo', kernel='linear')
            classifier.fit(training_input, training_output)
            prediction = classifier.predict(testing_input)

            scores.append(accuracy_score(testing_output, prediction))
        except:
            pass

    return scores

def calculate_normalized_cut(edges, node_community_membership, K):

    #Sanity check
    assert npmin(edges[:,0]) == 0

    #Make sure all nodes are not assigned to same community
    if unique(node_community_membership).shape[0] == 1:
        raise ValueError('There must be more than one community detected to calculate normalized cut')
    
    #Calculate normalized cut for each relation
    normalized_cuts = []
    for relation in unique(edges[:,1]):

        #Create directed graph so we can use networkx normalized cut function
        G = DiGraph()
        for edge in edges:
            if edge[1] == relation and edge[3] == 1:
                G.add_node(edge[0])
                G.add_node(edge[2])
                G.add_edge(edge[0], edge[2])

        #Calulate normalized cut for each community 
        for community in range(K):

            #Divide nodes with respect to the current community
            current_community = []
            other_communities = []
            for node in range(len(node_community_membership)):
                if node_community_membership[node] == community:
                    current_community.append(node)
                else:
                    other_communities.append(node)

            #Calculate normalized cut for current community
            if len(current_community) > 0 and len(other_communities) > 0:
                try:
                    normalized_cuts.append(normalized_cut_size(G,current_community,other_communities))
                except:
                    pass
    
    #Return average normalized cut        
    return sum(normalized_cuts) / len(normalized_cuts)

def calculate_conductance(edges, node_community_membership, K):
    
    #Make sure all nodes are not assigned to same community
    if unique(node_community_membership).shape[0] == 1:
        raise ValueError('There must be more than one community detected to calculate conductance')

    #Calculate conductance for each relation    
    conductances = []
    for relation in unique(edges[:,1]):

        #Create directed graph so we can use networkx conductance function
        G = DiGraph()
        for edge in edges:
            if edge[1] == relation and edge[3] == 1:
                G.add_node(edge[0])
                G.add_node(edge[2])
                G.add_edge(edge[0], edge[2])

        #Divide nodes with respect to the current community
        for community in range(K):
            current_community = []
            other_communities = []
            for node in range(len(node_community_membership)):
                if node_community_membership[node] == community:
                    current_community.append(node)
                else:
                    other_communities.append(node)

            #Calculate conductance for current community        
            if len(current_community) > 0 and len(other_communities) > 0:
                try:
                    conductances.append(conductance(G,current_community,other_communities))
                except:
                    pass

    #Return average conductance
    return sum(conductances) / len(conductances)

def save_embeddings(embedding_model, N, dataset, E):

    #Generate embeddings for all nodes
    embeddings = []
    for node in range(N):
        embedding = embedding_model.predict(array([node]))
        embeddings.append(squeeze(embedding))

    #Save embeddings
    save('MNE-Embeddings_Dataset-' + dataset + '_E-' + str(E), array(embeddings))

def save_community_memberships(community_membership_model, N, dataset, K):

    #Generate community memberships for all nodes
    community_memberships = []
    for node in range(N):
        community_membership = community_membership_model.predict(array([node]))
        community_memberships.append(community_membership)

    #Save community memberships
    save('MNE-Community-Memberships_Dataset-' + dataset + '_K-' + str(K), array(community_memberships))
