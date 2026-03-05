#!/usr/bin/env python3

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.
    
    nx: number of input features
    layers: list of nodes per layer
    activations: list of activation functions per layer
    lambtha: L2 regularization parameter
    keep_prob: probability a node will be kept (Dropout)
    """
    # 1. Create the conveyor belt (the model)
    model = K.Sequential()
    
    # 2. Define the 'Tax' (L2 Regularization)
    l2_reg = K.regularizers.l2(lambtha)
    
    # 3. Build each 'Station' (Layer)
    for i in range(len(layers)):
        if i == 0:
            # The very first layer needs to know the input size (nx)
            model.add(K.layers.Dense(
                layers[i], 
                activation=activations[i], 
                kernel_regularizer=l2_reg, 
                input_shape=(nx,)
            ))
        else:
            # Subsequent layers figure out their input size automatically
            model.add(K.layers.Dense(
                layers[i], 
                activation=activations[i], 
                kernel_regularizer=l2_reg
            ))
        
        # 4. Add 'Nap Time' (Dropout) after hidden layers
        # We don't add dropout after the final output layer (the last one)
        if i < len(layers) - 1:
            # Keras Dropout uses 'rate' (how many to drop)
            # rate = 1 - keep_probability
            model.add(K.layers.Dropout(1 - keep_prob))
            
    return model
