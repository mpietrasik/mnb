from keras.backend import squeeze
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import dot
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.models import Model

def initialize_model(A, E, K, N, R): 

    #Create subject and object input layers
    subject_input = Input(shape = (1,))
    object_input = Input(shape = (1,))

    #Create embedding layer which acts as a lookup table between input and embedding
    embedding_layer = Embedding(N, E, name='embedding_layer')

    #Pass inputs to embedding layer
    subject_embedding = embedding_layer(subject_input)
    object_embedding = embedding_layer(object_input)

    #Create softmax layer
    softmax_layer =Dense(K, activation = 'softmax', name='softmax_layer')

    #Pass embeddings through softmax layer, giving nodes' community membership distributions
    subject_community_membership = softmax_layer(subject_embedding)
    object_community_membership = softmax_layer(object_embedding)

    #Create community interactions tensor
    community_interactions_input = Input(shape = (1,))
    community_interactions_embedding_layer = Embedding(1, K * K * A)
    community_interactions = community_interactions_embedding_layer(community_interactions_input)
    community_interactions = Activation(activation='sigmoid', trainable = False)(community_interactions)
    community_interactions = Reshape((K, K, A))(community_interactions)

    #Index into community interactions tensor using community memberships
    community_interactions_vector = dot(inputs=[subject_community_membership, community_interactions], axes = (2, 1))
    community_interactions_vector = Reshape((K,A))(community_interactions_vector)
    object_community_membership_reshape = Reshape((K,1))(object_community_membership)
    community_interactions_vector = dot(inputs=[community_interactions_vector, object_community_membership_reshape], axes = 1)

    #Create relation input layer
    relation_input = Input(shape = (1,))

    #Pass relation inputs to embedding layer
    relation_embedding_layer = Embedding(R, A)
    relation_embedding = relation_embedding_layer(relation_input)
    relation_embedding = Reshape((A, 1))(relation_embedding)
    relation_embedding = Activation(activation='sigmoid', trainable = False)(relation_embedding)

    #Obtain interaction between subject and object on relation
    interaction = dot(inputs=[community_interactions_vector, relation_embedding], axes = 1)
    interaction = Lambda(lambda x: squeeze(x, 1))(interaction)

    #Compile main model
    model = Model(inputs=[subject_input, relation_input, object_input, community_interactions_input], outputs=interaction)
    model.compile(optimizer='adam', loss='mse')

    #Compile embedding model
    embedding_model = Model(inputs=subject_input, outputs=subject_embedding)

    #Compile community membership model
    community_membership_model = Model(inputs=subject_input, outputs=subject_community_membership)

    return (model, embedding_model, community_membership_model)