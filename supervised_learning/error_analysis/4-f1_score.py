#!/usr/bin/env python3
"""
Module: 4-f1_score
"""


def f1_score(confusion):
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    f1_score = 2 / ((1/sensitivity(confusion)) + (1/precision(confusion)))

    return f1_score
