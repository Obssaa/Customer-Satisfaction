from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging
class Evaluation(ABC):
    """
        Abstract class defining strategies for evaluaton of our models
    """
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """Calculates the scores for models.
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """Evaluation strategy thats uses Mean Sqauared Error."""

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        try:
            logging.info("Calculating MSE Score")
            mse = mean_squared_error(y_true= y_true, y_pred= y_pred)
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE{}".format(e))
            raise e


class R2(Evaluation):
     """Evaluation strategy thats uses R2."""
     
     def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray)-> float:
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            return r2
        except Exception as e:
            logging.error("Error in calculating R2{}".format(e))
            raise e