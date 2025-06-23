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
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse_class, acc_class, r2_score = evaluate_model(model,X_test,y_test)
    return  mse_class, acc_class, r2_score