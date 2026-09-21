#!/usr/bin/env python3
"""Defines the NST class for performing neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize an NST instance.

        Args:
            style_image (numpy.ndarray): image used as a style reference
            content_image (numpy.ndarray): image used as a content
                reference
            alpha (float): weight for the content cost
            beta (float): weight for the style cost
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescale an image so its largest side is 512 pixels and its pixel
        values are between 0 and 1.

        Args:
            image (numpy.ndarray): shape (h, w, 3), the image to scale

        Returns:
            tf.Tensor: shape (1, h_new, w_new, 3), the scaled image
        """
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]
        scale = 512 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(image, (new_h, new_w),
                                method=tf.image.ResizeMethod.BICUBIC)
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image
