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
        self.load_model()
        self.generate_features()

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

    def load_model(self):
        """
        Create the model used to calculate cost.

        The model uses VGG19 as a base, with max pooling layers replaced
        by average pooling layers. Its outputs are the outputs of the
        layers in style_layers followed by content_layer.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        x = vgg.input
        layer_outputs = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, name=layer.name)(x)
            else:
                x = layer(x)
            layer_outputs[layer.name] = x

        outputs = [layer_outputs[name] for name in self.style_layers]
        outputs.append(layer_outputs[self.content_layer])

        self.model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the gram matrix of a layer output.

        Args:
            input_layer (tf.Tensor or tf.Variable): shape (1, h, w, c),
                the layer output whose gram matrix should be calculated

        Returns:
            tf.Tensor: shape (1, c, c), the gram matrix of input_layer
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (-1, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)

        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """
        Extract the features used to calculate neural style cost.

        Sets the public instance attributes:
            gram_style_features: list of gram matrices calculated from the
                style layer outputs of the style image
            content_feature: the content layer output of the content image
        """
        preprocess = tf.keras.applications.vgg19.preprocess_input
        n_style = len(self.style_layers)

        style_outputs = self.model(preprocess(self.style_image * 255))
        content_outputs = self.model(preprocess(self.content_image * 255))

        self.gram_style_features = [self.gram_matrix(output)
                                    for output in style_outputs[:n_style]]
        self.content_feature = content_outputs[n_style]
