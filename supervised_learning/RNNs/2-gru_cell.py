#!/usr/bin/env python3
"""Defines a gated recurrent unit (GRU) cell"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: numpy.ndarray of shape (m, h) with the previous hidden state
        x_t: numpy.ndarray of shape (m, i) with the data input for the cell
        Returns: h_next, y
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.matmul(h_x, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(h_x, self.Wr) + self.br)

        rh_x = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(rh_x, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
