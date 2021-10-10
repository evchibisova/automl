""" """

import os
import sys
import pandas as pd
from sklearn.datasets import load_iris

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from automl.classification import SimpleAutoMLClassifier


if __name__ == "__main__":

    # load input data
    data = load_iris()
    X = data.data
    y = data.target

    # fit classifier
    classifier = SimpleAutoMLClassifier(verbose=True)
    classifier.fit(X, y)
    
    # predict on test vector
    result = classifier.predict(X[:1])
    print(f"\nResult prediction: {result}")

    # look at the selected model
    print(f"Selected model: {classifier.model}")
