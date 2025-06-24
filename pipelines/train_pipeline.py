from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig
@pipeline(enable_cache=True)
def train_pipeline(data_path:str, config: ModelNameConfig):
    """
    create the pipeline from steps
    """
    df = ingest_data(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test,config=config)
    mse_class, r2_score = evaluate_model(model,X_test,y_test)
    return  mse_class, r2_score