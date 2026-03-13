#!/usr/bin/env python3
"""
save and load config json
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Function: save_config
    """
    config_json = network.to_json()

    with open(filename, 'w') as f:
        f.write(config_json)

    return None

def load_config(filename):
    """
    Function: load_config
    """
    with open(filename, 'r') as f:
        config_json = f.read()

    return K.models.model_from_json(config_json)
