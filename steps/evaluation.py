import logging
import pandas as pd
from zenml import step
from src.evaluation import Evaluation, R2, MSE
from sklearn.base import RegressorMixin
from typing_extensions import Tuple
from typing import Annotated
from zenml.client import Client

import mlflow

expriment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=expriment_tracker.name)

def evaluate_model(model: RegressorMixin, X_test:pd.DataFrame, 
                   y_test:pd.Series) -> Tuple[
                       Annotated[float, "mse_class"],
                       Annotated[float, "r2_score"]
                   ]:
    """
    Evaluate the model on the ingested dataframe.
    Args:
        Model: RegressorMixin, 
        X_test:pd.DataFrame, 
        y_test:pd.Series
     Returns:

    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE().calculate_scores(y_test, prediction)
        mlflow.log_metric("mse",mse_class)
        r2_score =  R2().calculate_scores(y_test,prediction)
        mlflow.log_metric("r2",r2_score)
        return mse_class, r2_score
    except Exception as e:
        logging.error("Error in Evaluating model {}".format(e))