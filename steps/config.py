from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model config to select algorithm"""
    model_name: str = "LinearRegression"
