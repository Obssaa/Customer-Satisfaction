from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    """
    create the pipeline from steps
    """
    df = ingest_data(data_path=data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)

