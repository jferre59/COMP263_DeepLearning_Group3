"""
Module for determining best threshold and evaluating performance.

Usage in other modules:
from evaluate import evaluate, find_best_threshold
"""

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import numpy as np

def evaluate(model_name, y_true, y_prob, threshold=0.5):
    
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"Model: {model_name} (threshold: {threshold:.3f})")
    print(f"   PR-AUC: {pr_auc:.4f}   ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

    return pr_auc, roc_auc, y_pred

def find_best_threshold(y_true, y_prob, beta=2.0):
    """
    Finds the best threshold for classification based on F-beta score.
    beta=2.0 weighs recall over precision as detection of fraud more important
    than false positives.
    """

    print(f"Finding best threshold to maximize F-beta score (beta={beta})...")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = ((1 + beta**2) * precisions * recalls / 
              (beta**2 * precisions + recalls + 1e-9)) #Adds small value to avoid div by 0.
    
    best_idx = np.argmax(f_beta[:-1])

    return thresholds[best_idx], f_beta[best_idx]