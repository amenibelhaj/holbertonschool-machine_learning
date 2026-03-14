#!/usr/bin/env python3
"""
Module for saving and loading Keras models
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Args:
        network: the model to save
        filename: path of the file to save the model to
    
    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a file.

    Args:
        filename: path of the file to load the model from
    
    Returns:
        The loaded model
    """
    return K.models.load_model(filename)
