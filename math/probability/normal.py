#!/usr/bin/env python3
"""Defines the Normal class representing a normal distribution."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize a normal distribution.

        Args:
            data (list): data used to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = float(sum(data) / n)
            variance = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Args:
            x (float): the x-value

        Returns:
            float: the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.

        Args:
            z (float): the z-score

        Returns:
            float: the x-value of z
        """
        return self.mean + z * self.stddev
