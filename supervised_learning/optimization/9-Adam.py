#!/usr/bin/env python3
"""Updates variables using the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    Args:
        alpha: learning rate
        beta1: weight for the first moment
        beta2: weight for the second moment
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var
        s: previous second moment of var
        t: time step used for bias correction

    Returns:
        The updated variable, the new first moment, and the new second moment
    """
    # 1. Update moments
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # 2. Bias correction
    v_corrected = v_new / (1 - (beta1 ** t))
    s_corrected = s_new / (1 - (beta2 ** t))

    # 3. Update variable
    var_new = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var_new, v_new, s_new
