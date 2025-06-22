import logging
import pandas as pd
from zenml import step
from src.evaluation import Evaluation, Accuracy, R2, MSE
from sklearn.base import RegressorMixin
from typing_extensions import Tuple
from typing import Annotated

@step
def evaluate_model(model: RegressorMixin, X_test:pd.DataFrame, 
                   y_test:pd.Series) -> Tuple[
                       Annotated[float, "mse_class"],
                       Annotated[float, "acc_class"],
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
        mse_class = MSE()
        mse_class.calculate_scores(y_test=y_test, y_pred=prediction)
        acc_class = Accuracy()
        acc_class.calculate_scores(y_test=y_test, y_pred=prediction)
        r2_score =  R2()
        r2_score.calculate_scores(y_test=y_test, y_pred=prediction)
        return mse_class, acc_class, r2_score
    except Exception as e:
        logging.error("Error in Evaluating model {}".format(e))