#!/usr/bin/env python3
"""Valid Convolution module"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # 1. Calculate the dimensions of the output image
    nh = h - kh + 1
    nw = w - kw + 1

    # 2. Initialize the output array with zeros
    # Shape is (m, nh, nw) to accommodate all images
    convolved = np.zeros((m, nh, nw))

    # 3. Perform the convolution using only two loops (height and width)
    for i in range(nh):
        for j in range(nw):
            # Extract the slice (receptive field) from all images at once
            # Multiply by kernel and sum the result over the last two axes
            image_slice = images[:, i:i+kh, j:j+kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
