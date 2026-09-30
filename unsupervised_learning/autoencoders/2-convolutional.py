#!/usr/bin/env python3
"""Defines a function that creates a convolutional autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Create a convolutional autoencoder.

    Args:
        input_dims (tuple): dimensions of the model input
        filters (list): number of filters for each convolutional layer
            in the encoder, reversed for the decoder
        latent_dims (tuple): dimensions of the latent space
            representation

    Returns:
        tuple: (encoder, decoder, auto)
            encoder: the encoder model
            decoder: the decoder model
            auto: the full autoencoder model
    """
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), padding='valid',
                            activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    output = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same',
                                 activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, output)

    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
