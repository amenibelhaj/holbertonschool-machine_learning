#!/usr/bin/env python3
"""Module to build a Keras model using the Functional API"""
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
    # 1. Define the Entry Gate (Input Layer)
    inputs = K.Input(shape=(nx,))

    # 2. Define the 'Tax'
    reg = K.regularizers.l2(lambtha)

    # 3. Chain the layers
    # We start 'x' at the inputs
    x = inputs

    for i in range(len(layers)):
        # Create the Dense layer and immediately pass 'x' into it
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

        # Add Dropout after hidden layers (not the last one)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # 4. Create the final Model by connecting the start to the end
    model = K.Model(inputs=inputs, outputs=x)

    return model
