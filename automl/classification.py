""" """

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean


class SimpleAutoMLClassifier:
    """ 
    
    Args:
        metric (object, optional): (Default value = roc_auc_score)
    """

    def __init__(self, metric="roc_auc", verbose=False):
        #:
        self.metric = metric
        #:
        self.model = None
        #:
        self._estimated_models = [
            DecisionTreeClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier
            ]
        #:
        self.verbose = verbose

    def fit(self, X, y):
        """ 
        
        Args:
            X
            y
        """
        self.model = self._select_classifier(X, y)
        self.model.fit(X, y)

    def predict(self, X):
        """ 
        
        Args:
            X
        
        Returns:
            
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
