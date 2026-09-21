#!/usr/bin/env python3
"""Defines the Poisson class representing a Poisson distribution."""

e = 2.7182818285


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.

        Args:
            data (list): data used to estimate the distribution
            lambtha (float): expected number of occurrences in a given
                time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate the value of the PMF for a given number of successes.

        Args:
            k (int): number of successes

        Returns:
            float: the PMF value for k, or 0 if k is out of range
        """
        k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (self.lambtha ** k) * (e ** -self.lambtha) / factorial
