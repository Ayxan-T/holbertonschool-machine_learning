#!/usr/bin/env python3
"""
    Module: 1-from_dictionary
    This module is a part of 'Pandas' project.
"""
import pandas as pd

d = {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ['one', 'two', 'three', 'four']
}

df = pd.DataFrame(d, index=list("ABCD"))
