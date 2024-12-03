#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

np.random.seed(42)

##################################################################################
# Lab Class ML:III
# Starter code for exercise 6: Logistic Model for Generative Authorship Detection
##################################################################################

GROUP = "XX"  # TODO: write in your group number


def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features + 1).
    """
    # TODO: Your code here


def load_class_values(filename: str) -> np.array:
    """
    Load the class values for is_human (class 0 for False and class 1
    for True) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    # TODO: Your code here


def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        # TODO: Your code here


def logistic_function(w: np.array, x: np.array) -> float:
    """
    Return the output of a logistic function with parameter vector `w` on
    example `x`.
    Hint: use np.exp(np.clip(..., -30, 30)) instead of np.exp(...) to avoid
    divisions by zero
    """
    # TODO: Your code here


def logistic_prediction(w: np.array, x: np.array) -> float:
    """
    Making predictions based on the output of the logistic function
    """
    # TODO: Your code here


def initialize_random_weights(p: int) -> np.array:
    """
    Generate a pseudorandom weight vector of dimension p.
    """
    # TODO: Your code here


def logistic_loss(w: np.array, x: np.array, c: int) -> float:
    """
    Calculate the logistic loss function
    """
    # TODO: Your code here


def train_logistic_regression_with_bgd(xs: np.array, cs: np.array, eta: float=1e-8, iterations: int=2000, validation_fraction: float=0) -> Tuple[np.array, float, float]:
    """
    Fit a logistic regression model using the Batch Gradient Descent algorithm and
    return the learned weights as a numpy array.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values c(x) for every element in `xs` as a one-dimensional numpy array with length n
    - `eta`: the learning rate as a float value
    - `iterations': the number of iterations to run the algorithm for
    - 'validation_fraction': fraction of xs and cs used for validation (not for training)

    Returns:
    - the learned weights as a column vector, i.e. a two-dimensional numpy array with shape (1, p)
    - logistic loss value
    - misclassification rate of predictions on training part of xs/cs
    - misclassification rate of predictions on validation part of xs/cs
    """
    # TODO: Your code here


def plot_loss_and_misclassification_rates(loss: List[float],
                                          train_misclassification_rates: List[float],
                                          validation_misclassification_rates: List[float]):
    """
    Plots the normalized loss (divided by max(loss)) and both misclassification rates
    for each iteration.
    """
    # TODO: Your code here

########################################################################
# Tests
import os
from pytest import approx


def test_logistic_function():
    x = np.array([1, 1, 2])
    assert logistic_function(np.array([0, 0, 0]), x) == approx(0.5)
    assert logistic_function(np.array([1e2, 1e2, 1e2]), x) == approx(1)
    assert logistic_function(np.array([-1e2, -1e2, -1e2]), x) == approx(0)
    assert logistic_function(np.array([1e2, -1e2, 0]), x) == approx(0.5)


def test_bgd():
    xs = np.array([
        [1, -1],
        [1, 2],
        [1, -2],
    ])
    cs = np.array([0, 1, 0])

    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, 0.1, 100)
    assert w @ [1, -1] < 0 and w @ [1, 2] > 0
    w, _, _, _ = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100)
    assert w @ [1, -1] > 0 and w @ [1, 2] < 0



########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    train_features_file_name = sys.argv[1]
    train_classes_file_name = sys.argv[2]
    test_features_file_name = sys.argv[3]
    test_predictions_file_name = sys.argv[4]

    print("(a)")
    xs = load_feature_vectors(train_features_file_name)
    xs_test = load_feature_vectors(test_features_file_name)
    cs = load_class_values(train_classes_file_name)
    # TODO print number of examples with each class

    print("(b)")
    # TODO print misclassification rate of random classifier

    print("(c)")
    test_c_result = pytest.main(['-k', 'test_logistic_function', '--tb=short', __file__])
    if test_c_result != 0:
        sys.exit(test_c_result)
    print("Test logistic function successful")

    print("(d)")
    test_d_result = pytest.main(['-k', 'test_bgd', '--tb=short', __file__])
    if test_d_result != 0:
        sys.exit(test_d_result)
    print("Test bgd successful")
    w, loss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd(xs, cs, validation_fraction = 0.2)

    print("(e)")
    plot_loss_and_misclassification_rates(loss, train_misclassification_rates, validation_misclassification_rates)

    print("(f)")
    # TODO predict on test set and write to predictions-test.tsv
