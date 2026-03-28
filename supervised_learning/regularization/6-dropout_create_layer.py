#!/usr/bin/env python3
"""
Module to create a TensorFlow layer with Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a neural network layer in TensorFlow using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        the output of the new layer
    """
    # Use VarianceScaling to match the expected weight initialization
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg',
        distribution='uniform'
    )

    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    # Get the output from the Dense layer
    res = layer(prev)

    # Apply Dropout
    # rate in Keras is the probability of DROPPING (1 - keep_prob)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(res, training=training)
