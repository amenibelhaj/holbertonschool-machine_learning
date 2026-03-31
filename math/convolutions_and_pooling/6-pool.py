#!/usr/bin/env python3
"""Pooling module"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple (kh, kw)
        stride: tuple (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        A numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # 1. Calculate output dimensions
    # Pooling usually doesn't involve padding, so the formula is simpler
    nh = (h - kh) // sh + 1
    nw = (w - kw) // sw + 1

    # 2. Initialize output (Channels 'c' remain unchanged)
    pooled = np.zeros((m, nh, nw, c))

    # 3. Perform pooling with two loops
    for i in range(nh):
        for j in range(nw):
            v_start, v_end = i * sh, i * sh + kh
            h_start, h_end = j * sw, j * sw + kw

            # Select the window for ALL images and ALL channels at once
            # Shape: (m, kh, kw, c)
            image_slice = images[:, v_start:v_end, h_start:h_end, :]

            if mode == 'max':
                # Maximize over the height and width axes (1 and 2)
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                # Average over the height and width axes (1 and 2)
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
