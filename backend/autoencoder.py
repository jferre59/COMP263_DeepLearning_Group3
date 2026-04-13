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