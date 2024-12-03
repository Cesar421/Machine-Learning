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

GROUP = "16"


def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features + 1).
    """
    features = pd.read_csv(filename, delimiter="\t")                    # Load File
    vFeatures = features.iloc[:, 1:].values                             # Ignore 1st row of (Head of table)
    return np.array(vFeatures)


def load_class_values(filename: str) -> np.array:
    """
    Load the class values for is_human (class 0 for False and class 1
    for True) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    labels = pd.read_csv(filename, delimiter="\t")                      # Load File
    vLabels = labels.iloc[:, 1:].values                                 # Ignore 1st row of (Head of table)
    return np.array(vLabels)


def misclassification_rate(cs: np.array, ys: np.array) -> float:  
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """  
    sizeData = cs.size
    cs = np.transpose(cs)
    if len(cs) == 0:
        return float('nan')
    else:
        errors = np.sum(cs != ys)  # Directly count the number of mismatches
        mcr = errors/sizeData      # Calculate MCR with the number of errors
        return float(mcr)


def logistic_function(w: np.array, x: np.array) -> np.array:
    """
    Return the output of a logistic function with parameter vector `w` on
    example `x`.
    Hint: use np.exp(np.clip(..., -30, 30)) instead of np.exp(...) to avoid
    divisions by zero
    """
    w = np.transpose(w)                                     # Transpose W
    z = np.dot(x, w)
    z = np.array(z, dtype=np.float64)
    yLogistic = 1 / (1 + np.exp(np.clip(-z, -30, 30)))      # Logistic Regression Funct.
    return np.array(yLogistic)


def logistic_prediction(w: np.array, x: np.array) -> float:
    """
    Making predictions based on the output of the logistic function
    """
    predictions = logistic_function(w, x)
    classifications = np.array([True if value >= 0.5 else False for value in predictions])
    return classifications


def initialize_random_weights(p: int) -> np.array:
    """
    Generate a pseudorandom weight vector of dimension p.
    """
    weights = np.random.rand(1, p)
    return np.array(weights)


def logistic_loss(w: np.array, x: np.array, c: np.array) -> float:
    """
    Calculate the logistic loss function
    """
    numberElements = np.size(c)
    c_int = c.astype(int)
    yLogistic = logistic_function(w,x)                                  # Apply Logistic Function 

    lossArray = -c_int * np.log(yLogistic) - (1 - c_int) * np.log(1 - yLogistic)

    globalLoss = np.mean(lossArray)
    #globalLoss = (-1 / numberElements) * np.sum(lossArray)              # Obtain Global Loss
    return float(globalLoss)

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
    steps = range(iterations)
    sizeData = cs.size
    # Separate Validation and Train Sets
    idxValid = round((1-validation_fraction)*sizeData)
    xTrain = xs[:idxValid]
    xValid = xs[idxValid:]
    cTrain = cs[:idxValid]
    cValid = cs[idxValid:]

    # Initiation of Weights
    sizeX = xTrain.shape
    numElements = np.size(cTrain)
    w = initialize_random_weights(sizeX[1])
    # eta == alpha that is the learning rate
    # Need to update each step "W" and "X"

    # Store Loss
    trainLoss = []
    trainError = []
    validError = []
    wValues = []

    # Start iterative Steps (Training)
    for idx in steps:
        # Logistic function -> z = wT*x + wo
        yTrain = logistic_function(w,xTrain)
        
        # Calculation of derivatives of Weights(Gradient)
        difference = np.transpose(cTrain - yTrain)
        dW = (1 / numElements) * np.dot(difference,xTrain)
        # Update Weights (Move to local minima)
        w = w + eta * dW

        # Make predictions
        ysTrain = logistic_prediction(w,xTrain)
        ysValid = logistic_prediction(w,xValid)

        # Calculate Metrics with new Weights
        lossTrain = logistic_loss(w, xTrain, cTrain)
        missTrain = misclassification_rate(cTrain, ysTrain)
        missValid = misclassification_rate(cValid, ysValid)

        # Save Metrics
        trainLoss.append(lossTrain)
        trainError.append(missTrain)
        validError.append(missValid)
        wValues.append(w)

    resultIteration = (wValues,trainLoss,trainError, validError)
    y1 = resultIteration[0][-1]
    y2 = resultIteration[1][-1]
    y3 = resultIteration[2][-1]
    y4 = resultIteration[2][-1]

    yfinal = (y1, y2, y3, y4)

    return yfinal

def train_logistic_regression_with_bgd_List(xs: np.array, cs: np.array, eta: float =1e-8, iterations: int=2000, validation_fraction: float=0) -> Tuple[np.array, np.array, np.array]:
    steps = range(iterations)
    sizeData = cs.size
    # Separate Validation and Train Sets
    idxValid = round((1-validation_fraction)*sizeData)
    xTrain = xs[:idxValid]
    xValid = xs[idxValid:]
    cTrain = cs[:idxValid]
    cValid = cs[idxValid:]

    # Initiation of Weights
    sizeX = xTrain.shape
    numElements = np.size(cTrain)
    w = initialize_random_weights(sizeX[1])
    # eta == alpha that is the learning rate
    # Need to update each step "W" and "X"

    # Store Loss
    trainLoss = []
    trainError = []
    validError = []
    wValues = []

    # Start iterative Steps (Training)
    for idx in steps:
        # Logistic function -> z = wT*x + wo
        yTrain = logistic_function(w,xTrain)
        
        # Calculation of derivatives of Weights(Gradient)
        difference = np.transpose(cTrain - yTrain)
        dW = (1 / numElements) * np.dot(difference,xTrain)
        # Update Weights (Move to local minima)
        w = w + eta * dW

        # Make predictions
        ysTrain = logistic_prediction(w,xTrain)
        ysValid = logistic_prediction(w,xValid)

        # Calculate Metrics with new Weights
        lossTrain = logistic_loss(w, xTrain, cTrain)
        missTrain = misclassification_rate(cTrain, ysTrain)
        missValid = misclassification_rate(cValid, ysValid)

        # Save Metrics
        trainLoss.append(lossTrain)
        trainError.append(missTrain)
        validError.append(missValid)
        wValues.append(w)

    return (wValues,trainLoss,trainError, validError)

def plot_loss_and_misclassification_rates(loss: np.array,
                                          train_misclassification_rates: np.array,
                                          validation_misclassification_rates: np.array):
    """
    Plots the normalized loss (divided by max(loss)) and both misclassification rates
    for each iteration.
    """
    # Normalize the loss (put values between 0-1)
    maxLoss = np.max(loss)
    nLoss = loss / maxLoss

    # Create The Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Y Axis (Loss)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Normalized Loss', color='blue')
    ax1.plot(nLoss, label="Normalized Loss", color='blue', linestyle='-', marker='o', markevery=10)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.1)  # Set appropriate limits for normalized loss

    # Secondary Y Axis (missclasification Rate)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Misclassification Rates', color='red')
    ax2.plot(train_misclassification_rates, label="Training Misclassification Rate", color='red', linestyle='--', marker='x', markevery=100)
    ax2.plot(validation_misclassification_rates, label="Validation Misclassification Rate", color='green', linestyle='-.', marker='s', markevery=100)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 0.6)  # Adjust the range for better visualization

    # Combinated Legend
    fig.legend(loc='upper right')
    plt.title('Normalized Loss and Misclassification Rates')
    plt.grid(True)
    plt.show()

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
    cs = np.array([0, 1, 0]).reshape(-1, 1)

    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, 0.1, 100, 0)
    assert w @ np.array([1, -1]) < 0 and w @ np.array([1, 2]) > 0
    w, _, _, _ = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100, 0)
    assert w @ np.array([1, -1]) > 0 and w @ np.array([1, 2]) < 0



########################################################################
# Main program for running against the training dataset
# For visualization and Testing
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
    # Print number of examples with each class
    numElements = np.size(cs)
    class1 = np.sum(cs)  # Suma de 1s
    class0 = numElements - class1  # Los ceros son el total de elementos menos los 1s

    print(f"Examples of Class 1: {class1}")
    print(f"Examples of Class 0: {class0}")

    print("(b)")
    # Print misclassification rate of random classifier
    num_classes = np.unique(cs).shape[0]                                # Number of Class (2)
    yrand = np.random.choice(np.unique(cs), numElements)                # Do random Prediction
    yrand = yrand.reshape(-1, 1)
    missRandom = misclassification_rate(cs, yrand)
    print(f"Missclasification rate of Random Classifier is: {missRandom}")

    # The misclassification rate of a random classifier should be approx. the 50%, due random factor in a dataset that is binary
    # classified.


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
    w, loss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd_List(xs, cs, validation_fraction = 0.2)

    print("(e)")
    plot_loss_and_misclassification_rates(loss, train_misclassification_rates, validation_misclassification_rates)

    print("(f)")
    # Predict on test set and write to predictions-test.tsv
    w, loss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd(xs, cs, validation_fraction = 0.2)
    yPredict = logistic_prediction(w,xs_test)
    export = pd.DataFrame(yPredict, columns=['Predicted Class'])
    export.to_csv('predictions-test.tsv', sep='\t', index=False)

    #Loss and misclassification rate are related values but measure different aspects of model performance:
    # - The Loss captures how well the model predicts probabilitiesof classification.
    # - The Misclassification rate reflects only the proportion of incorrect predictions.
    # When the loss decreases, the model's predictions become more accurate and confident, often leading to a reduction in the misclassification rate.
    # but the model can have a low misclassification rate but still a high loss if it makes a few very confident but incorrect predictions.

    # Print confirmation
    print("(f) Predictions saved in predictions-test.tsv")
