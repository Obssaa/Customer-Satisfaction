import logging
import pandas as pd
from zenml import step

class IngestData():
    """
     Ingesting data from the data_path
    """
    def __init__(self, data_path:str):
        self.data_path = data_path

    
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}.")
        return pd.read_csv(self.data_path)
@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingesting data from the data_path
    
    Args: 
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingested_data = IngestData(data_path)
        df = ingested_data.get_data()
        return df
    
    except Exception as e:
        logging.error(f"Error while ingesting data.{e}")
        raise e
    

