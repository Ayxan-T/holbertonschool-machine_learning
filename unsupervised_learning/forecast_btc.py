#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import preprocess_data.preprocess_csvfile as preprocess


# Read preprocessed file
filepath = preprocess()
df = pd.read_csv(filepath)
