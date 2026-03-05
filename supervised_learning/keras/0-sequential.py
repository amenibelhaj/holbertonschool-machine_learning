#!/usr/bin/env python3
"""Module to build a Keras model"""
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
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            ))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
