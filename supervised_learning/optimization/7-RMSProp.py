#!/usr/bin/env python3
"""Updates variables using RMSProp optimization"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha: learning rate
        beta2: RMSProp weight (discounting factor)
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    # Calculate the new second moment (squared gradient moving average)
    s_new = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    # Update the variable
    # We divide the gradient by the root mean square of previous gradients
    var_new = var - (alpha * grad / (np.sqrt(s_new) + epsilon))

    return var_new, s_new
