"""
Module to build and train a classifier model that will take original transaction data
as well as autoencoder output to predict fraud.

Usage in other modules:
    from classifier import build_classifier, train_classifier
    clf = build_classifier(input_dim)
    history = train_classifier(clf, X_train_aug, y_train, X_val_aug, y_val)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_classifier(input_dim: int):
    """
    Shallow dense NN using BatchNorm & Dropout for regularization.
    Appends reconstruction error from autoencoder as col 0.
    """

    input = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(input, out, name="classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),   #focal loss for class imbalance
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )

    return model


def train_classifier(clf, X_train_aug, y_train, X_val_aug, y_val):

    print("\nTraining classifier model on augmented features...\n")

    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: fraud_ratio}
    print(f"Class weight for fraud: {fraud_ratio:.1f}x")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", patience=8,
                                      restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", factor=0.5,
                                          patience=4, mode="max"),
    ]

    history = clf.fit(
        X_train_aug, y_train,
        validation_data=(X_val_aug, y_val),
        epochs=120,
        batch_size=512,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    clf.save("model/clf.keras")
    print(f"\nClassifier trained for {len(history.history['loss'])} epochs.")

    return history