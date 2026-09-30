#!/usr/bin/env python3
"""Defines a function building a convolutional generator and
discriminator for face generation."""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Build a convolutional generator and discriminator.

    Returns:
        tuple: (generator, discriminator) Keras models
    """

    def get_generator():
        """Build the convolutional generator model."""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        for filters in [64, 16, 1]:
            x = keras.layers.UpSampling2D()(x)
            x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, x, name="generator")

    def get_discriminator():
        """Build the convolutional discriminator model."""
        inputs = keras.Input(shape=(16, 16, 1))
        x = inputs

        for filters in [32, 64, 128, 256]:
            x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
