#!/usr/bin/env python3
"""Strided Convolution module"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # 1. Determine Padding Values
    if padding == 'same':
        # Formula for 'same' padding to maintain input shape (if stride=1)
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # 2. Apply Padding
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                         mode='constant', constant_values=0)

    # 3. Calculate Output Dimensions
    nh = ((h + 2 * ph - kh) // sh) + 1
    nw = ((w + 2 * pw - kw) // sw) + 1

    # 4. Initialize Output
    convolved = np.zeros((m, nh, nw))

    # 5. Perform Convolution with Strides
    for i in range(nh):
        for j in range(nw):
            # Calculate the starting position based on stride
            v_start, v_end = i * sh, i * sh + kh
            h_start, h_end = j * sw, j * sw + kw
            
            image_slice = padded_imgs[:, v_start:v_end, h_start:h_end]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
