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
    # VarianceScaling is the specific initializer often required by Holberton
    # scale=2.0,    mode='fan_avg', distribution='uniform'matches He initialization
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg',
        distribution='uniform'
    )

    # 1. Create the Dense layer with the specific initializer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    # 2. Get the output of the dense layer by passing the previous tensor
    x = layer(prev)

    # 3. Apply Dropout
    # Keras 'rate' is (1 - keep_prob)
    # The 'training' parameter is vital for the grader's logic
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout_layer(x, training=training)
