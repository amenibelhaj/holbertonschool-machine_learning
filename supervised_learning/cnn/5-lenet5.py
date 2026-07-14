#!/usr/bin/env python3
"""Builds a modified LeNet-5 model"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture

    X: K.Input of shape (m, 28, 28, 1)

    Returns:
        compiled K.Model
    """

    initializer = K.initializers.HeNormal(seed=0)

    # Convolution layer 1
    Z1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # Max pooling layer 1
    P1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(Z1)

    # Convolution layer 2
    Z2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(P1)

    # Max pooling layer 2
    P2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(Z2)

    # Flatten
    F = K.layers.Flatten()(P2)

    # Fully connected layers
    FC1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(F)

    FC2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(FC1)

    # Output layer
    Y = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(FC2)

    model = K.Model(inputs=X, outputs=Y)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
