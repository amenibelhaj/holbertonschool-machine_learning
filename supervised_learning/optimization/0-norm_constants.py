#!/usr/bin/env python3
import numpy as np

def normalization_constants(X):
    """
    Calcule la moyenne (mean) et l'écart-type (std) de chaque colonne d'une matrice X.

    X : numpy.ndarray de forme (m, nx)
        m = nombre de points de données (lignes)
        nx = nombre de features (colonnes)

    Returns:
        mean : numpy.ndarray de forme (nx,) → moyenne de chaque colonne
        std  : numpy.ndarray de forme (nx,) → écart-type de chaque colonne
    """
    mean = X.mean(axis=0)  # moyenne colonne par colonne
    std = X.std(axis=0)    # écart-type colonne par colonne
    return mean, std
