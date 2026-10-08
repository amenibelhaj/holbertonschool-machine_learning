#!/usr/bin/env python3
"""
Creates, trains and validates a keras RNN model that uses the past
24 hours of BTC data to predict the BTC close price of the next hour

Usage: ./forecast_btc.py [preprocessed_npz]
(run ./preprocess_data.py first to create the npz file)
"""
import sys
import numpy as np
import tensorflow as tf


DATA = 'btc_preprocessed.npz'
MODEL_PATH = 'btc_forecast.keras'
WINDOW = 24
BATCH_SIZE = 64
EPOCHS = 50


def make_dataset(data, close_idx, shuffle=False):
    """
    Builds a tf.data.Dataset of sliding windows

    data: numpy.ndarray of shape (n, features), already standardized
    close_idx: index of the Close price in the features
    shuffle: whether to shuffle the windows (training set only)
    Returns: tf.data.Dataset yielding (inputs, target) batches
        inputs: tensor of shape (batch, WINDOW, features)
        target: tensor of shape (batch, 1), the close of the next hour
    """
    ds = tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
    ds = ds.window(WINDOW + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(WINDOW + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1, close_idx:close_idx + 1]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(n_features):
    """
    Creates the RNN model

    n_features: number of features per time step
    Returns: the compiled keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, n_features)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae'])
    return model


def main():
    """Trains, validates and tests the BTC forecasting model"""
    path = sys.argv[1] if len(sys.argv) > 1 else DATA
    npz = np.load(path)
    features = list(npz['features'])
    close_idx = features.index('Close')
    close_mean = npz['mean'][close_idx]
    close_std = npz['std'][close_idx]

    train_ds = make_dataset(npz['train'], close_idx, shuffle=True)
    val_ds = make_dataset(npz['val'], close_idx)
    test_ds = make_dataset(npz['test'], close_idx)

    model = build_model(len(features))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=[early_stop])

    test_mse, test_mae = model.evaluate(test_ds, verbose=0)
    print('Test MSE (standardized): {:.6f}'.format(test_mse))
    print('Test RMSE: {:.2f} USD'.format(np.sqrt(test_mse) * close_std))
    print('Test MAE: {:.2f} USD'.format(test_mae * close_std))

    last_day = npz['test'][-WINDOW:][np.newaxis].astype(np.float32)
    prediction = model.predict(last_day, verbose=0)[0, 0]
    print('Predicted close for the next hour: {:.2f} USD'.format(
        prediction * close_std + close_mean))

    model.save(MODEL_PATH)
    print('Model saved to {}'.format(MODEL_PATH))


if __name__ == '__main__':
    main()
