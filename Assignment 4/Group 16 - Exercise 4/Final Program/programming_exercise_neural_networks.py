#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

np.random.seed(42)

##################################################################
# Starter code for exercise 7: Neural Networks for Generative Authorship Detection
##################################################################

GROUP = "16"

# From last exercise sheet
def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features + 1).
    """
    features = pd.read_csv(filename, sep='\t').to_numpy()
    features[:, 0] = 1                                      # replace #id column with w0
    return features.astype(float)


# From last exercise sheet
def load_class_values(filename: str) -> np.array:
    """
    Load the class values for overall quality (class 0 for quality 1 and class 1
    for overall quality 2 or 3) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    return np.ravel((pd.read_csv(filename, sep='\t', usecols=["is_human"]).to_numpy() > 0) * 1)


def encode_class_values(cs: list[str], class_index: dict[str, int]) -> np.array:
    """
    Encode the given list of given class values as one-hot vectors.

    Arguments:
    - cs: a list of n class values from a dataset
    - class_index: a dictionary that maps each class value to a number between
         0 and k-1, where k is the number of distinct classes.

    Returns:
    - an array of shape (n, k) containing n column vectors with k elements each.
    """
    k = len(class_index)                                    # Posible Classification Options - #Change for 3 classes
    encodedCs = np.zeros((len(cs), k), dtype=int)           # Empty encoded List
    
    # Use this for to assign a value of 1 in points in which the class is correct
    for (idx, c) in enumerate(cs):
        encodedCs[idx, class_index[c]] = 1
    return encodedCs

def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        hits = [cs[i][ys[i]] for i in range(len(ys))]
        return 1 - (sum(hits) / len(ys))


# From code linked on lecture slide
def initialize_random_weights(p: int, l: int, k: int) -> Tuple[np.array, np.array]:
    """
    Initialize the weight matrices of a two-layer MLP.

    Arguments:
    - `p`: number of input attributes       (Attributes Ammount)
    - `l`: number of hidden layer features  
    - `k`: number of output classes         (Classes Values)

    Returns:
    - W_h, a l-by-(p+1) matrix
    - W_o, a k-by-(l+1) matrix
    """
    # the +1 is to consider the bias --> "wo"
    W_h = np.random.normal(size=(l, p+1))
    W_o = np.random.normal(size=(k, l+1))                       #Change for 3 classes
    return W_h, W_o


# From code linked on lecture slide / last exercise sheet
def sigmoid(z: np.array) -> np.array:
    return 1 / (1 + np.exp(np.clip(-z, -30, 30)))


# From code linked on lecture slide
def predict_probabilities(W_h: np.array, W_o: np.array, xs: np.array) -> np.array:
    """
    Predict the class probabilities for each example in xs.

    Arguments:
    - `W_h`: a l-by-(p+1) matrix
    - `W_o`: a k-by-(l+1) matrix
    - `xs`: feature vectors in the dataset as a two-dimensional numpy array
            with shape (n, p+1)

    Returns:
    - The probabilities for each of the k classes for each of the n examples as
      a two-dimensional numpy array with shape (n, k)
    """
    # Compute hidden layer activations and add y_0 = 1 (Bias)
    w_hT = np.transpose(W_h)                                     # Transpose W_hidden
    y_hidden = sigmoid(np.dot(xs, w_hT))
    y0_column = np.ones((y_hidden.shape[0], 1))                  # Create space for weight 0 -> Bias
    y_hidden = np.append(y0_column, y_hidden, axis=1)

    # Compute Output layer activation
    w_oT = np.transpose(W_o)                                     # Transpose W_Output
    y_output = sigmoid(np.dot(y_hidden, w_oT))

    # Use "Softmax" to obtain Probabilities of each class
    probabilities = np.exp(y_output) / np.sum(np.exp(y_output))  # Softmax #Change for 3 classes
    return probabilities

def predict(W_h: np.array, W_o: np.array, xs: np.array) -> np.array:
    """
    Predict the class for each example in xs.

    Arguments:
    - `W_h`: a l-by-(p+1) matrix
    - `W_o`: a k-by-(l+1) matrix
    - `xs`: feature vectors in the dataset as a two-dimensional numpy array
            with shape (n, p+1)

    Returns:
    - The predicted class for each of the n examples as an array of length n
    """
    prediction = predict_probabilities(W_h, W_o, xs)
    MaxPostion = np.argmax(prediction, axis=1)                  # The index of the position in row with the highest probability is marked
    return MaxPostion


# From code linked on lecture slide
def train_multilayer_perceptron(xs: np.array, cs: np.array, l: int, eta: float=0.0001, iterations: int=1000, validation_fraction: float=0) -> Tuple[list[Tuple[np.array, np.array]], list[float], list[float]]:
    """
    Fit a multilayer perceptron with two layers and return the learned weight matrices as numpy arrays.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values for every element in `xs` as a two-dimensional numpy array with shape (n, k)
    - `l`: the number of hidden layer features
    - `eta`: the learning rate as a float value
    - `iterations': the number of iterations to run the algorithm for
    - 'validation_fraction': fraction of xs and cs used for validation (not for training)

    Returns:
    - models (W_h, W_o) for each iteration, where W_h is a l-by-(p+1) matrix and W_o is a k-by-(l+1) matrix
    - misclassification rate of predictions on training part of xs/cs for each iteration
    - misclassification rate of predictions on validation part of xs/cs for each iteration
    """
    models = []
    train_misclassification_rates = []
    validation_misclassification_rates = []
    last_train_index = round((1 - validation_fraction) * len(cs))

    ## (1) Initialization
    p = len(xs[0]) - 1
    k = len(cs[0])  # Change for 3 classes
    W_h, W_o = initialize_random_weights(p, l, k)
    ## (2) Outer loop (over epochs/iterations)
    for t in range(iterations):
        # Update the learning rate at each iteration (Make it small each iteration)
        decay = 0.98                                                            #Between 0-1
        eta = eta * decay                                                       # Decay the learning rate
        
        ## (3) Inner loop (over training examples)
        for i in range(last_train_index):
            # (x as a column vector)
            x = np.reshape(xs[i], (len(xs[i]), 1))
            c = cs[i].reshape(k, 1)                                             #Change for 3 classes
    
            # (3-1) Forward propagation
            # Compute hidden layer activations and add y_0 = 1 (Bias)
            y_hidden = sigmoid(np.dot(W_h, x))
            y0_row = np.ones((1,y_hidden.shape[1]))                             # Create space for weight 0 -> Bias
            y_hidden = np.append(y0_row, y_hidden, axis=0)

            # Compute Output layer activation
            y_output = sigmoid(np.dot(W_o, y_hidden))
            
            # (3-2) Backpropagation
            # Deltas Calculation
            dError = y_output - c
            derivSigmoid = y_output * (1 - y_output)
            dWo = dError * derivSigmoid
            W_oT = np.transpose(W_o)                                            # Transpose W_Output
            dWh = np.dot(W_oT,dWo)[1:] * y_hidden[1:] * (1 - y_hidden[1:])      # Skip bias -> Dont modify position y_o(0) = 1
            
            # Gradient Calculation each weight
            xT = np.transpose(x)
            gWh = eta * np.dot(dWh, xT)
            y_hiddenT = np.transpose(y_hidden)
            gWo = eta * np.dot(dWo, y_hiddenT)
            
            # Update weights
            W_h = W_h - gWh
            W_o = W_o - gWo

        models.append((W_h.copy(), W_o.copy()))
        train_misclassification_rates.append(misclassification_rate(cs[0:last_train_index,:], predict(W_h, W_o, xs[0:last_train_index,:])))
        validation_misclassification_rates.append(misclassification_rate(cs[last_train_index:,:], predict(W_h, W_o, xs[last_train_index:,:])))
    return models, train_misclassification_rates, validation_misclassification_rates


# From last exercise sheet
def plot_misclassification_rates(train_misclassification_rates: List[float], validation_misclassification_rates: List[float]):
    """
    Plots both misclassification rates for each iteration.
    """
    plt.plot(train_misclassification_rates, label="Misclassification rate (train)")
    plt.plot(validation_misclassification_rates, label="Misclassification rate (validation)")
    plt.legend()
    plt.show()

########################################################################
# Tests
import os
from pytest import approx

def test_encode_class_values():
    cs = ['red', 'green', 'red', 'blue', 'green']
    class_index = {'red': 0, 'green': 1, 'blue': 2}

    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])

    actual = encode_class_values(cs, class_index)

    assert actual.shape == (5, 3), "encode_class_values should return array of shape (n, k)."

    assert actual.dtype == int, "encode_class_values should return an integer array."

    assert np.all(expected == actual), \
        "encode_class_values should return (n, k, 1)-array of one-hot vectors."

def test_predict_proabilities():
    class_index = {'red': 0, 'green': 1, 'blue': 2}
    cs = encode_class_values(['red', 'green', 'red', 'blue', 'green'], class_index)
    xs = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0.5],
        [1, 0, 0, 1],
        [1, 0, 1, 0.5]
    ])
    # Put out the x0 element for analysis -> xs start with a x0 this time
    p = len(xs[0]) - 1
    k = len(cs[0])
    W_h, W_o = initialize_random_weights(p, 8, k)

    probabilities = predict_probabilities(W_h, W_o, xs)
    assert probabilities.shape == (len(xs), k), \
        "predict_probabilities should return a shape of (n, k)"

def test_predict():
    class_index = {'red': 0, 'green': 1, 'blue': 2}
    cs = encode_class_values(['red', 'green', 'red', 'blue', 'green'], class_index)
    xs = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0.5],
        [1, 0, 0, 1],
        [1, 0, 1, 0.5]
    ])
    p = len(xs[0]) - 1
    k = len(cs[0])
    W_h, W_o = initialize_random_weights(p, 8, k)

    ys = predict(W_h, W_o, xs)
    assert ys.shape == (len(xs), ), \
        "predict should return a shape of (n, )"

def test_train():
    class_index = {'red': 0, 'green': 1, 'blue': 2}
    cs = encode_class_values(['red', 'green', 'red', 'blue', 'green'], class_index)
    xs = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0.5],
        [1, 0, 0, 1],
        [1, 0, 1, 0.5]
    ])
    models, _, _ = train_multilayer_perceptron(xs, cs, 2, eta=1, iterations=100, validation_fraction=0.4)
    W_h, W_o = models[-1] # get last model

    y = predict(W_h, W_o, np.array([[1, 1, 0, 0.2]]))
    assert y == class_index['red'], \
        "fit should learn a simple classification problem"



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

    xs = load_feature_vectors(train_features_file_name)
    xs_test = load_feature_vectors(test_features_file_name)

    print("(a)")
    test_a_result = pytest.main(['-k', 'test_encode_class_values', '--tb=short', __file__])
    if test_a_result != 0:
        sys.exit(test_a_result)
    print("Test encode_class_values function successful")

    # encode class "0" as [1 0] and class "1" as [0 1]
    class_index = {0: 0, 1: 1}
    cs = encode_class_values(load_class_values(train_classes_file_name), class_index)

    print("(b)")
    test_b_result = pytest.main(['-k', 'test_predict_proabilities', '--tb=short', __file__])
    if test_b_result != 0:
        sys.exit(test_b_result)
    print("Test predict_probabilities function successful")

    print("(c)")
    test_c_result = pytest.main(['-k', 'test_predict', '--tb=short', __file__])
    if test_c_result != 0:
        sys.exit(test_c_result)
    print("Test predict function successful")

    print("(d)")
    test_d_result = pytest.main(['-k', 'test_train', '--tb=short', __file__])
    if test_d_result != 0:
        sys.exit(test_d_result)
    print("Test train_multilayer_perceptron function successful")
    models, train_misclassification_rates, validation_misclassification_rates = train_multilayer_perceptron(xs, cs, 64, eta=0.001, iterations=800, validation_fraction=0.25)
    plot_misclassification_rates(train_misclassification_rates, validation_misclassification_rates)

    print("(e)")
    #How Obtain Best index - Search the one with less Validation Error
    idx_minvalidError = np.argmin(validation_misclassification_rates)
    print("Best Model was in index: ", idx_minvalidError)
    best_model_index = idx_minvalidError
    print("Minimal misclassification rate on validation set (index " + str(best_model_index) + "): " + str(validation_misclassification_rates[best_model_index]))
    W_h, W_o = models[best_model_index]
    y_test = predict(W_h, W_o, xs_test)
    np.savetxt(test_predictions_file_name, y_test, fmt='%d', delimiter='\t', newline='\n')


