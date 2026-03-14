#!/usr/bin/env python3
"""Module to convert labels to one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    labels: the vector of labels to convert
    classes: the total number of classes in the one-hot matrix

    Returns: the one-hot matrix
    """
    # Keras utility that handles the "box checking" logic
    # if classes is None, it will automatically find the highest number
    return K.utils.to_categorical(labels, num_classes=classes)
