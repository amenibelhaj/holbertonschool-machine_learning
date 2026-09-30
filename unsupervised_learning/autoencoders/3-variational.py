#!/usr/bin/env python3
"""Defines a function that creates a variational autoencoder."""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder.

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in
            the encoder, reversed for the decoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        tuple: (encoder, decoder, auto)
            encoder: the encoder model, outputting the latent
                representation, the mean and the log variance
            decoder: the decoder model
            auto: the full autoencoder model
    """
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sample a latent vector using the reparameterization trick."""
        mu, log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(encoder_input, [z, mean, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, output)

    auto_input = keras.Input(shape=(input_dims,))
    encoded, mu, log_sigma = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)

    def vae_loss(y_true, y_pred):
        """Compute reconstruction loss plus KL divergence."""
        reconstruction = keras.backend.binary_crossentropy(y_true,
                                                           y_pred)
        reconstruction = keras.backend.sum(reconstruction, axis=1)
        kl = 1 + log_sigma - keras.backend.square(mu)
        kl = kl - keras.backend.exp(log_sigma)
        kl = -0.5 * keras.backend.sum(kl, axis=1)
        return reconstruction + kl

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
