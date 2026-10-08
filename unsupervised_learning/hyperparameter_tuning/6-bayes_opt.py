#!/usr/bin/env python3
"""Optimizes a neural network classifier with Bayesian optimization
using GPyOpt."""
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

data = load_digits()
X = data.data / 16.0
Y = data.target
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=0)

bounds = [
    {'name': 'learning_rate', 'type': 'continuous',
     'domain': (1e-4, 1e-1)},
    {'name': 'units', 'type': 'discrete',
     'domain': (16, 32, 64, 128, 256)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.6)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (16, 32, 64, 128)},
]


def train_model(params):
    """Train one model and return its best validation accuracy."""
    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2 = float(params[0][3])
    batch_size = int(params[0][4])

    name = 'lr{:.5f}_units{}_dropout{:.3f}_l2{:.6f}_batch{}.h5'.format(
        learning_rate, units, dropout, l2, batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(units, activation='relu',
                              kernel_regularizer=
                              tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        name, monitor='val_accuracy', save_best_only=True)

    history = model.fit(X_train, Y_train, epochs=100,
                        batch_size=batch_size,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[early_stop, checkpoint], verbose=0)

    accuracy = max(history.history['val_accuracy'])
    tf.keras.backend.clear_session()

    return -accuracy


if __name__ == '__main__':
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=train_model, domain=bounds, acquisition_type='EI',
        maximize=False)
    optimizer.run_optimization(max_iter=30)

    optimizer.plot_convergence()
    plt.savefig('convergence.png')

    best = optimizer.x_opt
    with open('bayes_opt.txt', 'w') as f:
        f.write('Bayesian optimization report\n')
        f.write('============================\n\n')
        f.write('Model: 1 hidden layer neural network on the digits '
                'dataset\n')
        f.write('Satisficing metric: validation accuracy\n')
        f.write('Iterations: 30\n\n')
        f.write('Best hyperparameters:\n')
        for i, param in enumerate(bounds):
            f.write('  {}: {}\n'.format(param['name'], best[i]))
        f.write('\nBest validation accuracy: {}\n'.format(-optimizer.fx_opt))
