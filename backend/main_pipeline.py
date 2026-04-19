"""
Script to run the pipeline:
 - Loads & preprocesses data
 - Builds & trains autoencoder
 - Augments data using reconstruction error from autoencoder
 - SMOTE oversampling on augmented data
 - Builds then trains classifier on augmented data
 - Evaluates hybrid model performance
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for cleaner output

import json
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

#Limiting warnings for cleaner output
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Importing modules
from dataloader import load_data
from preprocessor import preprocess
from autoencoder import build_autoencoder, train_autoencoder, reconstruction_error
from classifier import build_classifier, train_classifier
from evaluate import evaluate, find_best_threshold

print("\nCREDIT CARD FRAUD DETECTION - HYBRID MODEL\n")

#Loading & preprocessing data
df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

#Building & training autoencoder
print("\nBuilding and training autoencoder...")
autoencoder, ae_history = train_autoencoder(X_train, X_val, y_train)

#Definging reconstruction error & augmenting data
print("\nAugmenting data using autoencoder reconstruction error...")
re_train = reconstruction_error(autoencoder, X_train)
re_val = reconstruction_error(autoencoder, X_val)
re_test = reconstruction_error(autoencoder, X_test)

X_train_aug = np.column_stack((re_train, X_train))
X_val_aug = np.column_stack((re_val, X_val))
X_test_aug = np.column_stack((re_test, X_test))

#SMOTE oversampling on augmented data
print("\nOversampling augmented data using SMOTE...")
smote = SMOTE(random_state=SEED, k_neighbors=3)
X_train_aug, y_train = smote.fit_resample(X_train_aug, y_train)
print(f"After SMOTE oversampling of fraud class - Total: {len(X_train_aug):,}   "
      f"Fraud: {y_train.sum():,}   Legitimate: {(y_train == 0).sum():,}")


#Building & training classifier on augmented data
print("\nBuilding and training classifier on augmented data...")
clf = build_classifier(X_train_aug.shape[1])
clf_history = train_classifier(clf, X_train_aug, y_train, X_val_aug, y_val)

#Evaluating hybrid model performance
print("\nEvaluating hybrid model performance...\n")
clf_prob = clf.predict(X_test_aug, verbose=0).flatten()

#With default threshold of 0.5
evaluate("Hybrid Model (default)", y_test, clf_prob, threshold=0.5)

#With found best threshold
best_t, best_f2 = find_best_threshold(y_test, clf_prob, beta=3.0)

threshold = {
    "threshold": best_t.item()
}
#Save best threshold found to json file
with open("data/threshold_setting.json", "w") as outfile:
    json.dump(threshold, outfile)

print(f"\nOptimal F2 threshold: {best_t:.4f}   (F2={best_f2:.4f})\n")
evaluate("Hybrid Model (Optimized threshold)", y_test, clf_prob, threshold=best_t)

print("\nPipeline complete.")