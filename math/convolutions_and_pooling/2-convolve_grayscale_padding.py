#!/usr/bin/env python3
"""Convolution with Padding module"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # 1. Apply zero padding to the height and width axes
    # (0, 0) for the 'm' axis (don't pad number of images)
    # (ph, ph) for the height, (pw, pw) for the width
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # 2. Calculate the new dimensions of the output image
    nh = h + (2 * ph) - kh + 1
    nw = w + (2 * pw) - kw + 1

    # 3. Initialize output
    convolved = np.zeros((m, nh, nw))

    # 4. Perform the convolution using two loops
    for i in range(nh):
        for j in range(nw):
            # Slice from the PADDED images
            image_slice = padded_images[:, i:i+kh, j:j+kw]
            # Element-wise multiply and sum across the kernel dimensions
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
