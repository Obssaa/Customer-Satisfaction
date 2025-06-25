from pipelines.train_pipeline import train_pipeline
from steps.config import ModelNameConfig  
from zenml.client import Client

if __name__ == "__main__":
    train_pipeline(
        data_path="data/olist_customers_dataset.csv",
        config=ModelNameConfig(model_name="LinearRegression") 
    )
    print(Client().active_stack.experiment_tracker.get_tracking_uri())