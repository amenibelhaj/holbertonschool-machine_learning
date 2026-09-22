#!/usr/bin/env python3
"""Defines the BayesianOptimization class that performs Bayesian
optimization on a noiseless 1D Gaussian process."""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian
    process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize Bayesian optimization.

        Args:
            f: the black-box function to be optimized
            X_init (numpy.ndarray): shape (t, 1), the inputs already
                sampled with the black-box function
            Y_init (numpy.ndarray): shape (t, 1), the outputs of the
                black-box function for each input in X_init
            bounds (tuple): (min, max) bounds of the search space
            ac_samples (int): number of samples analyzed during
                acquisition
            l (float): length parameter for the kernel
            sigma_f (float): standard deviation given to the output of
                the black-box function
            xsi (float): exploration-exploitation factor for acquisition
            minimize (bool): whether optimization should be performed for
                minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        minimum, maximum = bounds
        self.X_s = np.linspace(minimum, maximum, ac_samples)
        self.X_s = self.X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
