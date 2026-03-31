#!/usr/bin/env python3
"""Convolution with Channels module"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # 1. Determine Padding Values
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # 2. Apply Padding
    # Note: We only pad the height and width (axes 1 and 2)
    # We do NOT pad the 'm' (axis 0) or 'c' (axis 3) dimensions
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)

    # 3. Calculate Output Dimensions
    nh = ((h + 2 * ph - kh) // sh) + 1
    nw = ((w + 2 * pw - kw) // sw) + 1

    # 4. Initialize Output (Shape: m, nh, nw)
    convolved = np.zeros((m, nh, nw))

    # 5. Perform Convolution using two loops
    for i in range(nh):
        for j in range(nw):
            v_start, v_end = i * sh, i * sh + kh
            h_start, h_end = j * sw, j * sw + kw

            # image_slice shape: (m, kh, kw, c)
            image_slice = padded_imgs[:, v_start:v_end, h_start:h_end, :]

            # Element-wise multiply across all m images and all c channels
            # Then sum over the height, width, and channel axes (1, 2, 3)
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return convolved
