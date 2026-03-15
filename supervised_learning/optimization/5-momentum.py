#!/usr/bin/env python3
"""Updates variables using Gradient Descent with Momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum algorithm

    Args:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    # Calculate the new first moment (velocity)
    # v_new = beta1 * v_prev + (1 - beta1) * current_gradient
    v_new = (beta1 * v) + ((1 - beta1) * grad)

    # Update the variable using the learning rate and the new velocity
    var_new = var - (alpha * v_new)

    return var_new, v_new
