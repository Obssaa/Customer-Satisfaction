import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivisionStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(data:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    ]:
    """Cleans and Divides data into train and test
    Args: 
        data:raw data
    Returns:
        X_train: Training Data
        X_test : Testing Data
        y_tran: Training Labels
        y_test: Testing Labels
    """
    try:
          process_strategy = DataPreProcessStrategy ()
          data_cleaning = DataCleaning (data, process_strategy)
          processed_data= data_cleaning.handle_data()
          divide_strategy =  DataDivisionStrategy()
          data_cleaning = DataCleaning (processed_data, divide_strategy)
          X_train, X_test, y_train, y_test = data_cleaning. handle_data ()
          
          logging.info("Data cleaning completed.")
          return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error("Error in cleaning data".format(e))
        raise e