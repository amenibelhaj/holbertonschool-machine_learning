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
