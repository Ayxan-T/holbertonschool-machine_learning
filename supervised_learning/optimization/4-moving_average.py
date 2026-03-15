#!/usr/bin/env python3
"""
Module: 
"""

import numpy as np


def moving_average(data, beta):
    v = 0  # Initialize the moving average at 0
    ma_corrected = []
    
    for t in range(1, len(data) + 1):
        # 1. Standard EWMA update (The "Raw" value)
        v = (beta * v) + (1 - beta) * data[t-1]
        
        # 2. Calculate Bias Correction factor
        # This gets closer to 1 as t increases
        correction_factor = 1 - (beta**t)
        
        # 3. Apply correction ONLY for the output
        v_corrected = v / correction_factor
        ma_corrected.append(v_corrected)
        
    return ma_corrected
