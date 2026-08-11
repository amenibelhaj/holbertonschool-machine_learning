#!/usr/bin/env python3
"""Identity Block Module"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block for a ResNet

    Args:
        A_prev: output from the previous layer
        filters: tuple/list containing
            F11 - filters for first 1x1 convolution
            F3 - filters for 3x3 convolution
            F12 - filters for second 1x1 convolution

    Returns:
        The activated output of the identity block
    """

    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=0)

    # First component
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer
    )(A_prev)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation("relu")(X)

    # Second component
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer
    )(X)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation("relu")(X)

    # Third component
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer
    )(X)

    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut connection
    X = K.layers.Add()([X, A_prev])

    # Final activation
    X = K.layers.Activation("relu")(X)

    return X
