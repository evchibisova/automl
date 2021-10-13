"""AutoML classifier module."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean


class SimpleAutoMLClassifier:
    """
    AutoML classifier with several integrated scikit-learn models.

    Provides usual sklearn model interface - uses `fit` method for learning and `predict` for making predictions.
    Automatically selects the best ML-model on defined metric (ROC AUC by default)
    
    Args:
        metric (str|object, optional): Scikit-learn metric, string or callable (Default value = 'roc_auc')
        verbose (bool): True for showing intermediate fit results, False otherwise.
    """

    def __init__(self, metric="roc_auc", verbose=False):
        self.metric = metric
        self.verbose = verbose
        #: Fitted model with best metric, initalised after `fit` method running. 
        self.model = None
        #: The list of estimated ML models.
        self._estimated_models = [
            DecisionTreeClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier
            ]
        
    def fit(self, X, y, verbose=False):
        """ 
        Model training method.
        Takes input train & test folds, cross-validate inner models, selects the best one, and fits it.
        
        Args:
            X (array-like): Training data.
            y (array-like): Target values.
        """
        self.model = self._select_classifier(X, y)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Method for making prediction using trained model.
        
        Args:
            X (array-like): Input data.
        
        Returns:
            array: Predicted values.
        """
        return self.model.predict(X)
        
    def _select_classifier(self, X, y):
        best_classifier = self._estimated_models[0]
        best_score = 0

        for model_cls in self._estimated_models:
            model = model_cls()
            score = mean(cross_val_score(model, X, y))

            if self.verbose:
                print(f"{str(model)} score: \t{score}")

            if score > best_score:
                best_score = score
                best_classifier = model

        return best_classifier
