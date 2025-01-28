#!/usr/bin/env python3
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from math import prod
############################################################################
# Starter code for exercise: Naive Bayes for Generative Authorship Detection
############################################################################

GROUP = "16"


def load_bow_feature_vectors(filename: str) -> np.array:
    """
    Load the Bag-of-words feature vectors from the given file and return
    them as a two-dimensional numpy array with shape (n, p), where n is the number
    of examples in the dataset and p is the number of features per example.
    """
    return np.load(filename)

# From last exercise sheet
def load_class_values(filename: str) -> np.array:
    """
    Load the class values from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
     # Load the data, specifying the 'is_human' column
    data = pd.read_csv(filename, sep='\t', usecols=["is_human"])
    
    # Convert the 'is_human' column to a numpy array and return
    return np.ravel((data.to_numpy() > 0) * 1)


def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        hits = np.sum(cs == ys)
        return 1 - hits / len(cs)
    


def class_priors(cs: np.ndarray) -> dict:
    """Compute the prior probabilities P(C=c) for all the distinct classes c in the given dataset.

    Args:
        cs (np.ndarray): one-dimensional array of values c(x) for all examples x from the dataset D

    Returns:
        dict: a dictionary mapping each distinct class to its prior probability
    """
     # Convert class labels to integers if they are not already
    unique_classes, cs_int = np.unique(cs, return_inverse=True)
    
    # Ensure cs_int is a one-dimensional array
    cs_int = np.ravel(cs_int)

    # Count the occurrences of each class
    class_counts = np.bincount(cs_int)

    # Total number of examples
    total_count = len(cs_int)

    # Compute prior probabilities
    priors = {unique_classes[c]: count / total_count for c, count in enumerate(class_counts) if count > 0}
    return priors

def extract_features(filename: str) -> np.array:
    """
    Load the TSV file from the given filename and return a numpy array with the
    features extracted from the text.
    """
    # Load the data
    data = pd.read_csv(filename, sep='\t', header = None, names = ['text'])

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features = 5000, # use a maximum number of features to reduce dimensionality
        ngram_range = (1, 2), # Capture both unigrams and bigrams
        stop_words = 'english', # Remove common stop words
        lowercase = True, # Convert all text to lower case
        max_df = 0.95, # Ignore terms appearing in more than 95% of documents
        min_df = 0.01 # Ignore terms appearing in less than 1% of documents
    )

    # Fit and transform the text data into TF-IDF features
    features = vectorizer.fit_transform(data['text'])

    # Convert the sparse matrix to a dense numpy array
    return features.toarray()
 


def conditional_probabilities(xs: np.ndarray, cs: np.ndarray) -> dict:
    """Compute the conditional probabilities P(B_j = x_j | C = c) for all combinations of feature B_j, feature value x_j and class c found in the given dataset.

    Args:
        xs (np.ndarray): n-by-p array with n points of p attributes each
        cs (np.ndarray): one-dimensional n-element array with values c(x)

    Returns:
        dict: nested dictionary d with d[c][B_j][x_j] = P(B_j = x_j | C=c)
    """
    # Initialize the nested dictionary using defauktdict
    conditional_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    #Identify unique classes
    unique_classes = np.unique(cs)

    #Iterate over each class
    for c in unique_classes:
        #Extract examples belonging to class c
        class_indicies = (cs == c)
        class_examples = xs[class_indicies]

        #Count the total number of examples for class c
        total_class_examples = class_examples.shape[0]

        #Iterate  over each feature index (column)
        for j in range(xs.shape[1]):
            # Count occurences of each feature value in column j
            feature_values, counts = np.unique(class_examples[:, j], return_counts = True)

            #Calculate conditional probabilities for each feature value
            for value, count in zip(feature_values, counts):
                conditional_probs[c][j][value] = count / total_class_examples
    return conditional_probs



class NaiveBayesClassifier:
    def fit(self, xs: np.ndarray, cs: np.ndarray):
        """Fit a Naive Bayes model on the given dataset

        Args:
            xs (np.ndarray): n-by-p array of feature vectors
            cs (np.ndarray): n-element array of class values
        """
       # Convert class labels to integers and store the mapping
        self.unique_classes, cs_int = np.unique(cs, return_inverse=True)
        
        # Check the shapes of xs and cs_int
        print(f"Shape of xs: {xs.shape}")
        print(f"Length of cs_int: {len(cs_int)}")

        # Ensure the number of samples matches
        if xs.shape[0] != len(cs_int):
            raise ValueError("Mismatch between number of samples in features and class labels.")

        # Calculate the prior probabilities for the class
        self.priors = class_priors(cs_int)

        # Calculate the conditional probabilities for the features given the class
        self.conditional_probs = conditional_probabilities(xs, cs_int)


    def predict(self, x: np.ndarray) -> str:
        """Generate a prediction for the data point x

        Args:
            x (np.ndarray): a p-dimensional feature vector

        Returns:
            str: the most probable class for x
        """
        max_class_index = None
        max_probability = -float('inf')  # Initialize with negative infinity for comparison

        # Iterate over each class in the priors
        for c_index, prior_prob in self.priors.items():
            # Start with the prior probability (in log form to avoid numerical underflow)
            log_prob = np.log(prior_prob)

            # Multiply by the conditional probabilities for each feature
            for j, x_j in enumerate(x):
                # Use a small smoothing value (e.g., 1e-9) for missing feature values
                conditional_prob = self.conditional_probs[c_index][j].get(x_j, 1e-9)
                log_prob += np.log(conditional_prob)
            
            # Track the class with the maximum posterior probability
            if log_prob > max_probability:
                max_probability = log_prob
                max_class_index = c_index
        
        # Return the original class label
        return self.unique_classes[max_class_index]
        


def train_and_predict(training_features_file_name: str, training_labels_file_name: str, 
                      validation_features_file_name: str, validation_labels_file_name: str,
                      test_features_file_name: str) -> np.ndarray:
    """Train a model on the given training dataset, and predict the class values
    for the given testing dataset. Report the misclassification rate on the training
    and validation sets.

    Return an array with the predicted class values, in the same order as the
    examples in the testing dataset.
    """
    # Load the datasets
    xs_train = np.load(training_features_file_name)
    cs_train = load_class_values(training_labels_file_name)
    
    # Ensure the number of labels matches the number of feature vectors
    if len(cs_train) > xs_train.shape[0]:
        cs_train = cs_train[:xs_train.shape[0]]
    
    xs_val = np.load(validation_features_file_name)
    cs_val = load_class_values(validation_labels_file_name)
    
    # Ensure the number of labels matches the number of feature vectors
    if len(cs_val) > xs_val.shape[0]:
        cs_val = cs_val[:xs_val.shape[0]]
    
    xs_test = np.load(test_features_file_name)

    # Initialize the classifier
    nb_classifier = NaiveBayesClassifier()

    # Train the classifier on the training data
    nb_classifier.fit(xs_train, cs_train)

    # Predict the training set labels
    train_predictions = np.array([nb_classifier.predict(x) for x in xs_train])
    train_misclassification_rate = np.mean(train_predictions != cs_train)

    # Predict the validation set labels
    val_predictions = np.array([nb_classifier.predict(x) for x in xs_val])
    val_misclassification_rate = np.mean(val_predictions != cs_val)

    # Print misclassification rates
    print(f"Training Misclassification Rate: {train_misclassification_rate: .4f}")
    print(f"Validation Misclassification Rate: {val_misclassification_rate: .4f}")

    # Predict the test labels
    test_predictions = np.array([nb_classifier.predict(x) for x in xs_test])

    return test_predictions

########################################################################
# Tests
import os
from pytest import approx

train_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-train.npy')
train_classes_file_name = os.path.join(os.path.dirname(__file__), 'data/labels-train.tsv')
val_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-val.npy')
val_classes_file_name = os.path.join(os.path.dirname(__file__), 'data/labels-val.tsv')
test_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-test.npy')

def test_that_the_group_name_is_there():
    import re
    assert re.match(r'^[0-9]{1,3}$', GROUP), \
        "Please write your group name in the variable at the top of the file!"

def test_that_training_features_are_here():
    assert os.path.isfile(train_features_file_name), \
        "Please put the training dataset file next to this script!"

def test_that_training_classes_are_here():
    assert os.path.isfile(train_classes_file_name), \
        "Please put the validation dataset file next to this script!"
    
def test_that_validation_features_are_here():
    assert os.path.isfile(val_features_file_name), \
        "Please put the validation dataset file next to this script!"

def test_that_validation_classes_are_here():
    assert os.path.isfile(val_classes_file_name), \
        "Please put the validation dataset file next to this script!"

def test_that_test_features_are_here():
    assert os.path.isfile(test_features_file_name), \
        "Please put the test dataset file next to this script!"

def test_class_priors():
    cs = np.array(list('abcababa'))
    priors = class_priors(cs)
    assert priors == dict(a=0.5, b=0.375, c=0.125)

def test_conditional_probabilities():
    cs = np.array(list('aabb'))
    xs = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0]
    ])

    p = conditional_probabilities(xs, cs)

    assert p['a'][0][1] == 0.5
    assert p['a'][0][0] == 0.5
    assert p['b'][0][2] == 1
    assert p['a'][1][0] == 0.5
    assert p['a'][1][1] == 0.5
    assert p['b'][1][0] == 0.5
    assert p['b'][1][1] == 0.5
    assert p['a'][2][1] == 0.5
    assert p['a'][2][0] == 0.5
    assert p['b'][2][0] == 1

### example dataset from the lecture
xs_example = np.array([x.split() for x in """sunny hot high weak
sunny hot high strong
overcast hot high weak
rain mild high weak
rain cold normal weak
rain cold normal strong
overcast cold normal strong
sunny mild high weak
sunny cold normal weak
rain mild normal weak
sunny mild normal strong
overcast mild high strong
overcast hot normal weak
rain mild high strong""".split('\n')])

cs_example = np.array("no no yes yes yes no yes no yes yes yes yes yes no".split())

def test_classifier():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny cold high strong'.split()))
    assert pred == 'no', 'should classify example from the lecture correctly'

def test_classifier_unknown_value():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny hot dry none'.split()))
    assert pred == 'no', 'should handle unknown feature values'


########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict.")
    preds = train_and_predict(train_features_file_name, train_classes_file_name,
                              val_features_file_name, val_classes_file_name,
                              test_features_file_name)
    if preds is not None:
        print("Saving predictions.")
        pd.DataFrame(preds).to_csv(f"naive-bayes-predictions-test-group-{GROUP}.tsv", header=False, index=False, sep='\t')