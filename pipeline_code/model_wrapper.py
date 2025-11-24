from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from pipeline_code.model_tools import video_train_test_split
from pipeline_code.model_tools import undersample
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib as job
from sklearn.pipeline import Pipeline

class ModelWrapper:

    def __init__(self, X : pd.DataFrame,
                 y : pd.DataFrame,
                 model : BaseEstimator,
                 train_test_test_videos : int = 3,
                 random_state = None,
                 scaling : bool = True,
                 undersampling : bool = False):
        
        self.identity = "WrappedModel"
        self.meta = {}
        self.model = model
        self.features = X.columns
        self.train_index, self.test_index = video_train_test_split(X, y, test_videos = train_test_test_videos, random_state = random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X.loc[self.train_index],X.loc[self.test_index],y.loc[self.train_index],y.loc[self.test_index]
        self.meta["has_DataFrame"] = True
        if scaling:
            num_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            self.X_train[num_features] = scaler.fit_transform(self.X_train[num_features])
            self.X_test[num_features] = scaler.transform(self.X_test[num_features])
            self.meta["scaler"] = scaler
        
        if undersampling:
            rus = RandomUnderSampler(random_state=random_state)
            rus.fit_resample(self.X_train,self.y_train)
            self.meta["undersampler"] = rus

    @classmethod
    def load(cls, input_path : str, X : pd.DataFrame = None, y : pd.DataFrame = None):

        object = job.load(input_path)
        if X != None and y != None:
            object.X_train, object.X_test, object.y_train, object.y_test = X.loc[object.train_index],X.loc[object.test_index],y.loc[object.train_index].values.ravel(),y.loc[object.test_index].values.ravel()
            #columns in loaded dataframe
            object.meta["has_DataFrame"] = True
        else:
            print("Object was instanced without DataFrame")
            object.meta["has_DataFrame"] = False
        return object
    
    def __str__(self):
        return f"{self.identity} for: {self.model} with {len(self.features)} input features\n train set '{self.train_index}'\n test set '{self.test_index}'\n modifications: {self.meta}"
    
    def fit(self):
        raveled_y_train = self.y_train.values.ravel()
        self.model.fit(X = self.X_train, y = raveled_y_train)
        self.meta["fitted_on"] = self.train_index

    def predict(self):
        predictions_dictionary = {}
        for video_id in self.test_index:
            print(f"running prediction for video {video_id}")
            prediction = self.model.predict(self.X_test.loc[video_id])
            predictions_dictionary[video_id] = pd.Series(prediction)
        self.y_pred = pd.DataFrame(pd.concat(predictions_dictionary.values(), keys = predictions_dictionary.keys(), names = ['video_id', 'frame']))
        self.meta["prediction"] = {"y_pred" : self.y_pred, "y_true" : self.y_test,"video_id" : self.test_index}
        return self.y_pred

    def save(self, output_path : str):
        print("saving...")
        del self.X_train, self.X_test, self.y_train, self.y_test
        job.dump(self, output_path)
        print("model saved")
    


class GridWrapper(ModelWrapper):

    def __init__(self, X : pd.DataFrame,
                 y : pd.DataFrame,
                 estimator : Pipeline | BaseEstimator,
                 param_grid : dict,
                 scoring : str = "f1_macro",
                 cv : int = 5,
                 n_jobs : int = -1,
                 train_test_test_videos : int = 3,
                 random_state = None,
                 scaling : bool = True,
                 undersampling : bool = False):
        
        super().__init__(X = X,
                 y = y,
                 model = None,  # We don't have the best estimator yet
                 train_test_test_videos = train_test_test_videos,
                 random_state = random_state,
                 scaling = scaling,
                 undersampling = undersampling)
        
        self.grid = GridSearchCV(estimator = estimator,
            param_grid = param_grid,
            scoring = scoring,     
            cv = cv,                 
            verbose = 2,
            n_jobs = n_jobs             
        )
        self.identity = "WrappedGrid"

    def search(self):
        raveled_y_train = self.y_train.values.ravel()
        self.grid.fit(y = raveled_y_train,X = self.X_train)
        self.model = self.grid.best_estimator_ # Now we have the best estimator
        self.hyperparameters = self.grid.best_params_
        self.meta["Grid_results"] = {"model" : self.model, "hyperparamters" : self.hyperparameters}
    
    def get_wrapped_model(self):

        wrappedmodel = ModelWrapper(
            X = self.X,
            y = self.y,
            model = self.model,  
            train_test_test_videos = self.train_test_test_videos,
            random_state = self.random_state,
            scaling = self.scaling,
            undersampling = self.undersampling)
        
        return wrappedmodel
        

    
