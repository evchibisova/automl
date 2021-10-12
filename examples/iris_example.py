"""An example module on `iris` dataset"""

import os
import sys
import pandas as pd
from sklearn.datasets import load_iris

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simple_automl.classification import SimpleAutoMLClassifier


if __name__ == "__main__":

    # load input data
    data = load_iris()

    # fit classifier
    classifier = SimpleAutoMLClassifier(verbose=True)
    print("Test models and count metrics:")
    classifier.fit(data.data, data.target)

    # look at the selected model
    print(f"\nSelected model: {classifier.model}")
    
    # predict on test data
    test_data = data.data[:1]
    result = classifier.predict(test_data)
    print(f"\nPredict on input data: {test_data}")
    print(f"Result prediction: {result}")

