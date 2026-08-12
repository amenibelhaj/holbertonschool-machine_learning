from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture."""
    
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution
    x = K.layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(inputs)

    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # Max pooling
    x = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(x)

    # Stage 2
    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Stage 3
    x = projection_block(x, [128, 128, 512], s=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # Stage 4
    x = projection_block(x, [256, 256, 1024], s=2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    # Stage 5
    x = projection_block(x, [512, 512, 2048], s=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # Average pooling
    x = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(x)

    # Output layer
    x = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    return K.Model(inputs=inputs, outputs=x)
