from zenml import steps, pipeline
import some_ml_libraries

@steps
def data_preparation():
    data = some_ml_libraries.load_data()
    return data
@steps
def feature_engineering(data):
    features = some_ml_libraries.extract_features(data)
    return features
@steps
def model_training(features):
    model = some_ml_libraries.train_model(features)
@steps
def model_evaluation(model,features):
    result = some_ml_libraries.evaluate_model(model, features)
    return result
@steps
def model_deployment(model):
    some_ml_libraries.deploy(model)
@pipeline
def movie_production_pipeline():
    data = data_preparation()
    features = feature_engineering(data)
    model = model_training(features)
    evaluation = model_evaluation(model,features)
    model_deployment(model)
if __name__ == "__main__":
    movie_production_pipeline()