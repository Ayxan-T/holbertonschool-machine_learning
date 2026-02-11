#!/usr/bin/env python3

import numpy as np


def posterior(x, n, P, Pr):
    
    # Validation of inputs (YOU CAN IGNORE THIS PART) 
    if type(n) is not int or n <= 0: 
        raise ValueError("n must be a positive integer") 
    if type(x) is not int or x < 0: 
        raise ValueError("x must be an " "integer that is greater than or equal to 0") 
    if x > n: 
        raise ValueError("x cannot be greater than n") 
    if type(P) is not np.ndarray or P.ndim != 1: 
        raise TypeError("P must be a 1D numpy.ndarray") 
    if type(Pr) is not np.ndarray or Pr.shape != P.shape: 
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for elm in P:
        if elm < 0 or elm > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for elm in Pr:
        if elm < 0 or elm > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    ######################################################

    """
    The Problem/Situation: 
        You are conducting a study on a revolutionary cancer drug.
        During your trials, n patients take the drug and x patients
        develop severe side effects. 

        You can assume that x follows a binomial distribution.
        Each element in P is a possible probability value for a single trial
        (patient having developed side effects).
        And each i'th element in Pr is a probability that a single trial
        has P[i] chance of developing severe side effects.

    What this script does:
        Given that x out of n trials were successful, this script
        calculates the prior probability of having P[i] probability
        that a trial develops severe side effects.

    Main formula:
        Posterior = Likelihood * Prior / Marginal
        P(p|x,n) = P(x,n|p) * P(p) / P(x,n)

            where p => P[i] and P(p) => Pr[i]
            (P and Pr are numpy arrays)
    """
    
    # Calculating Likelihood
    combinations = np.math.factorial(n) / \
        (np.math.factorial(n - x) * np.math.factorial(x))
    likelihoods = combinations * (P**x) * ((1 - P)**(n - x))

    # Applying prior probabilities ( Intersection = Likelihood * Prior )
    intersection = likelihoods * Pr

    marginal_prob = np.sum(intersection)

    posterior = intersection / marginal_prob

    return posterior
