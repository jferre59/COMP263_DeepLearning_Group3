"""
Module for building & training autoencoder.
Usage in other modules:
    from autoencoder import build_autoencoder, train_autoencoder, reconstruction_error
    autoencoder, ae_history = train_autoencoder(X_train, X_val, y_train)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_autoencoder(input_dim: int):
    """
    Encoder:  input -> 64 -> 32 -> 16
    Decoder:  16 -> 32 -> 64 -> input
    L2 regularisation + Dropout to avoid over-fitting to noise.
    """

    input = keras.Input(shape=(input_dim,))

        # Encoder
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(input)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    encoded = layers.Dense(16, activation="relu", name="encoder")(x)

    # Decoder
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.Dense(64, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="linear", name="decoder")(x)

    autoencoder = keras.Model(input, decoded, name="Autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mse")
    return autoencoder

def train_autoencoder(X_train, X_val, y_train):
    """
    Training autoencoder on legitimate transactions ONLY.
    """

    print("Training autoencoder on legitimate transactions...")

    X_legit_train = X_train[y_train == 0]
    X_legit_val = X_val

    autoencoder = build_autoencoder(X_train.shape[1])

    callbacks =  [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]

    history = autoencoder.fit(
        X_legit_train, X_legit_train,
        epochs=120,
        batch_size=256,
        validation_data=(X_legit_val, X_legit_val),
        callbacks=callbacks,
        verbose=0
    )

    autoencoder.save("model/autoencoder.keras")
    print(f"\nAutoencoder trained for {len(history.history['loss'])} epochs.")

    return autoencoder, history


def reconstruction_error(autoencoder, X):
    """
    Compute MSE between input and reconstructed output.
    """
    X_pred = autoencoder.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)