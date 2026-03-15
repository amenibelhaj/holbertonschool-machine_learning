#!/usr/bin/env python3
"""Creates a learning rate decay operation in tensorflow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow 
    using inverse time decay

    Args:
        alpha: the original learning rate
        decay_rate: weight used to determine the rate of decay
        decay_step: number of passes before alpha is decayed further

    Returns:
        The learning rate decay operation (a schedule object)
    """
    # InverseTimeDecay calculates: 
    # initial_learning_rate / (1 + decay_rate * step / decay_step)
    # staircase=True makes the decay occur in a stepwise fashion (floor division)
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
