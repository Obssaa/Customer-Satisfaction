import json

# from .utils import get_data_for_test
import os

import numpy as np
import pandas as pd
# from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.config import ModelNameConfig  
# from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])
import pandas as pd

@step
def deployment_trigger(
    accuracy:float,
    min_accuracy: float = 0.92  
) -> bool:
    """Implements a simple model deployment trigger that looks at the input model accuracy"""
    return accuracy >= min_accuracy

@pipeline(enable_cache=True, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    data_path:str,
    min_accuracy:float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test,config=ModelNameConfig(model_name="LinearRegression"))
    mse_class, r2_score = evaluate_model(model,X_test,y_test)
    deployment_decision = deployment_trigger(r2_score, min_accuracy = min_accuracy)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers  = workers,
        timeout = timeout
    )