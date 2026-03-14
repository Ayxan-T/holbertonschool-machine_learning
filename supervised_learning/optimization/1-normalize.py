#!/usr/bin/env python3
"""
Module: 1-normalize
"""

import numpy as np
import tensorflow.keras as K


def normalize(X, m, s):
  """
  for X<d, nx>, mean m, stdev s,
  returns normalized X
  """
  return (x - m) / s
