#!/usr/bin/env python3
"""Updates the learning rate using inverse time decay in numpy"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion

    Args:
        alpha: the original learning rate
        decay_rate: weight used to determine the rate of decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes before alpha is decayed further

    Returns:
        The updated value for alpha
    """
    # Stepwise decay: we use floor division (//) to determine
    # how many full 'decay_step' intervals have passed.
    intervals_passed = global_step // decay_step
    
    # Apply the inverse time decay formula
    alpha_updated = alpha / (1 + decay_rate * intervals_passed)
    
    return alpha_updated
