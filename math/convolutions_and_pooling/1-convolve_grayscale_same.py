#!/usr/bin/env python3
"""Same convolution for grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # padding
    ph = kh // 2
    pw = kw // 2

    # pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # output
    output = np.zeros((m, h, w))

    # convolution
    for i in range(h):
        for j in range(w):
            slice_img = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(
                slice_img * kernel,
                axis=(1, 2)
            )

    return output
