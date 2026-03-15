#!/usr/bin/env python3
"""Sets up RMSProp optimizer in TensorFlow"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight (discounting factor)
        epsilon: small number to avoid division by zero

    Returns:
        The optimizer object
    """
    # In Keras/TF, beta2 is passed as the 'rho' parameter
    return tf.keras.optimizers.RMSProp(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
