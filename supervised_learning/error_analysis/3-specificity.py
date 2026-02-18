#!/usr/bin/env python3
"""
Module: 3-specifity
"""

import numpy as np


def specificity(confusion):
    """
    Function: specifity
    """
    total_samples = np.sum(confusion)

    actuals_total = np.sum(confusion, axis=1)

    predicteds_total = np.sum(confusion, axis=0)

    # specifity = TN / (TN + FP)

    tp = np.diag(confusion)

    fp = predicteds_total - tp

    tn = total_samples - actuals_total - (predicteds_total - tp)

    specifities = tn / (tn + fp)

    return specifities
