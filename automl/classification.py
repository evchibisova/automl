""" """

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


class SimpleAutoMLClassifier:
    """ 
    
    Args:
        metric (object, optional): (Default value = roc_auc_score)
    """

    def __init__(self, metric=roc_auc_score):
        #:
        self.model = None
        #:
        self.metric = metric
        #:
        self._models_variants = [
            DecisionTreeClassifier, 
            LogisticRegression,
            RandomForestClassifier,
            GradientBoostingClassifier
            ]

    def fit(self, X, y):
        """ 
        
        Args:
            X
            y
        """

        self.model = self._select_classifier()
        self.model.fit(X, y)

    def predict(self, X):
        """ 
        
        Args:
            X
        
        Returns:
            
        
        """
        return self.model.predict(X)
        
    def _select_classifier(self):
        best_classifier = DecisionTreeClassifier()
        return best_classifier
        
