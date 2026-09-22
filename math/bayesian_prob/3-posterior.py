#!/usr/bin/env python3
"""Defines a function that calculates a posterior probability."""


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining the data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
        numpy.ndarray: 1D array of the likelihood of obtaining the data
            for each probability in P
    """
    combinations = 1
    for i in range(1, x + 1):
        combinations = combinations * (n - x + i) // i

    return combinations * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining the data with the various
    hypothetical probabilities.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities
        Pr (numpy.ndarray): 1D array of the prior beliefs of P

    Returns:
        numpy.ndarray: 1D array of the intersection of obtaining x and n
            with each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of obtaining the data.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities
        Pr (numpy.ndarray): 1D array of the prior beliefs about P

    Returns:
        float: the marginal probability of obtaining x and n
    """
    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities
        Pr (numpy.ndarray): 1D array of the prior beliefs of P

    Returns:
        numpy.ndarray: the posterior probability of each probability in P
            given x and n
    """
    inter = intersection(x, n, P, Pr)

    return inter / np.sum(inter)
