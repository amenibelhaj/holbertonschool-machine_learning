#!/usr/bin/env python3
"""
Module to calculate the cost of a Keras model with L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost without L2 regularization
        model: Keras model with layers including L2 regularization

    Returns:
        A tensor containing the total cost for each layer,
        accounting for L2 regularization
    """
    # model.losses contains the regularization losses for each layer
    # We add the base cost to each of these individual penalties
    return cost + model.losses
