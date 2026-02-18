#!/usr/bin/env python3
"""
Module: 0-create_confusion
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Function: create_confusion_matrix
    """

    conf_mat = np.zeros((labels.shape[1], labels.shape[1]), dtype=float)

    true_vals = np.argmax(labels, axis=1)
    pred_vals = np.argmax(logits, axis=1)

    for t, p, in zip(true_vals, pred_vals):
        conf_mat[t, p] += 1

    return conf_mat
