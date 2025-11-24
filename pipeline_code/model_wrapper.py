from sklearn.linear_model import LinearRegression
import pandas as pd
from model_tools import video_train_test_split
from model_tools import undersample

class ModelWrapper:

    def __init__(self, X : pd.DataFrame , y : pd.DataFrame, model : LinearRegression, train_test_random_seed = None):
        self.model = model
        self.features = X.columns
        self.train_video_ids = None
        self.test_video_ids = None



    
    def __str__(self):
        return f"Wrapper object for model: {self.model} with {len(self.features)} input features"