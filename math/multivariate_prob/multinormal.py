#!/usr/bin/env python3
"""Defines the MultiNormal class representing a Multivariate Normal
distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initialize a Multivariate Normal distribution.

        Args:
            data (numpy.ndarray): shape (d, n), the data set
                n: number of data points
                d: number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean
        self.cov = (centered @ centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculate the PDF at a data point.

        Args:
            x (numpy.ndarray): shape (d, 1), the data point whose PDF
                should be calculated

        Returns:
            float: the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        exponent = -0.5 * (diff.T @ inv @ diff)

        return float(coefficient * np.exp(exponent[0][0]))
