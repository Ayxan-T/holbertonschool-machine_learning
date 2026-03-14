#!/usr/bin/env python3
"""
Module: 1-normalize
"""

import numpy as np


def normalize(X, m, s):
  """
  for X<d, nx>, mean m, stdev s,
  returns normalized X
  """
  return (x - m) / s
