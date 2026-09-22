#!/usr/bin/env python3
"""Defines the GaussianProcess class representing a noiseless 1D
Gaussian process."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init,
                 l=1, sigma_f=1):
        """
        Initialize a Gaussian process.

        Args:
            X_init (numpy.ndarray): shape (t, 1), the inputs already
                sampled with the black-box function
            Y_init (numpy.ndarray): shape (t, 1), the outputs of the
                black-box function for each input in X_init
            l (float): length parameter for the kernel
            sigma_f (float): standard deviation given to the output of
                the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l, self.sigma_f = l, sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices using
        the Radial Basis Function (RBF).

        Args:
            X1 (numpy.ndarray): shape (m, 1)
            X2 (numpy.ndarray): shape (n, 1)

        Returns:
            numpy.ndarray: shape (m, n), the covariance kernel matrix
        """
        a = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        b = np.sum(X2 ** 2, axis=1)
        sq_dist = a + b - 2 * X1 @ X2.T

        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sq_dist)

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of points in a Gaussian
        process.

        Args:
            X_s (numpy.ndarray): shape (s, 1), the points whose mean and
                standard deviation should be calculated

        Returns:
            tuple: (mu, sigma)
                mu: numpy.ndarray of shape (s,), the mean for each point
                sigma: numpy.ndarray of shape (s,), the variance for
                    each point
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        cov = K_ss - K_s.T @ K_inv @ K_s

        return mu.reshape(-1), np.diagonal(cov)

    def update(self, X_new, Y_new):
        """
        Update a Gaussian process with a new sample point.

        Args:
            X_new (numpy.ndarray): shape (1,), the new sample point
            Y_new (numpy.ndarray): shape (1,), the new sample function
                value
        """
        self.X = np.append(self.X, X_new[:, np.newaxis], axis=0)
        self.Y = np.append(self.Y, Y_new[:, np.newaxis], axis=0)
        self.K = self.kernel(self.X, self.X)
