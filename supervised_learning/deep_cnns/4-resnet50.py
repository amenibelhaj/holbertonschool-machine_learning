#!/usr/bin/env python3

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture."""

    inputs = K.Input(shape=(224, 224, 3))
