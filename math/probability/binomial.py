#!/usr/bin/env python3
"""Defines the Binomial class representing a binomial distribution."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize a binomial distribution.

        Args:
            data (list): data used to estimate the distribution
            n (int): number of Bernoulli trials
            p (float): probability of a success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - variance / mean
            self.n = round(mean / p)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculate the value of the PMF for a given number of successes.

        Args:
            k (int): number of successes

        Returns:
            float: the PMF value for k, or 0 if k is out of range
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(m):
            """Return the factorial of m."""
            result = 1
            for i in range(1, m + 1):
                result *= i
            return result

        combinations = factorial(self.n) // (factorial(k)
                                             * factorial(self.n - k))

        return combinations * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculate the value of the CDF for a given number of successes.

        Args:
            k (int): number of successes

        Returns:
            float: the CDF value for k, or 0 if k is out of range
        """
        k = int(k)
        if k < 0:
            return 0

        total = 0
        for i in range(k + 1):
            total += self.pmf(i)

        return total
