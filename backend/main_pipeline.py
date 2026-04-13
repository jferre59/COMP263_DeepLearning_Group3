"""
Script to run the pipeline:
 - Loads & preprocesses data
 - Builds & trains autoencoder
 - Augments data using reconstruction error from autoencoder
 - SMOTE oversampling on augmented data
 - Builds then trains classifier on augmented data
 - Evaluates hybrid model performance
"""

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Importing modules
from dataloader import load_data
from preprocessor import preprocess
from autoencoder import build_autoencoder, train_autoencoder, reconstruction_error
from classifier import build_classifier, train_classifier
from evaluate import evaluate, find_best_threshold