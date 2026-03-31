#!/usr/bin/env python3
"""Multiple Kernels Convolution module"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)

    # 3. Calculate Output Dimensions
    nh = ((h + 2 * ph - kh) // sh) + 1
    nw = ((w + 2 * pw - kw) // sw) + 1

    # 4. Initialize Output (Shape: m, nh, nw, nc)
    convolved = np.zeros((m, nh, nw, nc))

    # 5. Perform Convolution using three loops
    for i in range(nh):
        for j in range(nw):
            for k in range(nc):
                v_start, v_end = i * sh, i * sh + kh
                h_start, h_end = j * sw, j * sw + kw

                # Get the slice of all images
                image_slice = padded_imgs[:, v_start:v_end, h_start:h_end, :]

                # Multiply by the k-th kernel and sum across axes (1, 2, 3)
                # result shape is (m,)
                convolved[:, i, j, k] = np.sum(image_slice *
                                               kernels[:, :, :, k],
                                               axis=(1, 2, 3))

    return convolved
