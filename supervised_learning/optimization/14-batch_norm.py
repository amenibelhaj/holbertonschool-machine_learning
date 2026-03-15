#!/usr/bin/env python3
"""Creates a batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be used on the output
        
    Returns:
        A tensor of the activated output for the layer
    """
    # 1. Define the Kernel Initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Create the Dense layer (Linear Transformation)
    # We set use_bias=False because Batch Norm's 'beta' handles the offset
    model_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = model_layer(prev)

    # 3. Create and apply the Batch Normalization layer
    # gamma_initializer='ones' and beta_initializer='zeros' are defaults
    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7
    )
    Z_norm = batch_norm(Z)

    # 4. Apply the activation function
    if activation is None:
        return Z_norm
    return activation(Z_norm)
