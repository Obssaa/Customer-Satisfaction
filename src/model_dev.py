import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
class Model(ABC):
    """Abstract Class for all Models"""

    @abstractmethod
    def train(self, X_train:pd.DataFrame, y_train:pd.Series):
        """
        Trains the model
        Args:
            X_train: Training data 
            y_train: Training labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """Class for the Linear Regressoin Model"""
    def train(self, X_train, y_train, **kwargs):
        try:
            regression = LinearRegression(**kwargs)
            regression.fit(X_train,y_train)
            logging.info("Model training completed.")
            return regression
        except Exception as e:
            logging.error("Error while training the model{}".format(e))
            raise e