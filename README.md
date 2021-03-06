# simple_automl

Python automl library.

Fit classifier with your data - library will automatically choose the best scikit-learn ML-algorithm for prediction.


## Installation

No package installation for this test project - just make your virtual environment with requirements.txt, e.g.:

```bash
python -m venv automl-env
source automl-env/bin/activate
pip install -r requirements.txt
```

## Examples

Run `python examples/iris_example.py` to see simple example on `iris` dataset.


## Basic usage

Fit base classifier using input feature matrix and target:

```python
from simple_automl.classification import SimpleAutoMLClassifier

classifier = SimpleAutoMLClassifier()
classifier.fit(X_train, y_train)
```

Make prediction on input matrix:

```python
classifier.predict(X)
```

By default `SimpleAutoMLClassifier` selects the most appropriate model using cross-validation with ROC AUC metric. You can change it with `metric` parameter (use sklearn [metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)):

```python
classifier = SimpleAutoMLClassifier(metric="accuracy")
```

For seeing metrics for each tested model, use `verbose` flag:

```python
classifier = SimpleAutoMLClassifier(metric="accuracy", verbose=True)
```
