#!/usr/bin/env python3
"""Projection block."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block."""
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=0)

    # Main path
    conv1 = K.layers.Conv2D(
        F11,
        (1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)

    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        kernel_initializer=initializer
    )(act1)

    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(
        F12,
        (1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act2)

    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Shortcut path
    shortcut = K.layers.Conv2D(
        F12,
        (1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)

    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add the main path and shortcut
    add = K.layers.Add()([bn3, shortcut])

    # Final activation
    return K.layers.Activation('relu')(add)
