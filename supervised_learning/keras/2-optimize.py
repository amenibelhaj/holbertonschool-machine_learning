#!/usr/bin/env python3
"""Module to set up Adam optimization for a Keras model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics.

    network: the model to optimize
    alpha: the learning rate
    beta1: the first Adam optimization parameter
    beta2: the second Adam optimization parameter

    Returns: None
    """
    # 1. Initialize the Adam Optimizer with the given parameters
    opt = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # 2. "Compile" the model - this connects the brain to the learning strategy
    network.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return None
