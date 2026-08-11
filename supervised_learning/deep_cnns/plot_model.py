#!/usr/bin/env python3

from tensorflow import keras as K
from importlib import import_module

identity_block = import_module("2-identity_block").identity_block

X = K.Input(shape=(224, 224, 256))
Y = identity_block(X, [64, 64, 256])

model = K.Model(inputs=X, outputs=Y)

K.utils.plot_model(
    model,
    to_file="identity_block.png",
    show_shapes=True,
    show_layer_names=True
)

print("Done")
