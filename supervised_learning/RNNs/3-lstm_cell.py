#!/usr/bin/env python3
"""Defines a long short-term memory (LSTM) cell"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: numpy.ndarray of shape (m, h) with the previous hidden state
        c_prev: numpy.ndarray of shape (m, h) with the previous cell state
        x_t: numpy.ndarray of shape (m, i) with the data input for the cell
        Returns: h_next, c_next, y
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(h_x, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(h_x, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(h_x, self.Wc) + self.bc)
        o = self.sigmoid(np.matmul(h_x, self.Wo) + self.bo)

        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
