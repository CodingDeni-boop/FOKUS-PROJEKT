from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from pipeline_code.model_tools import video_train_test_split
from pipeline_code.model_tools import iter_predict
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib as job
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelWrapper:

    def __init__(self, X : pd.DataFrame,
                 y : pd.DataFrame,
                 model : BaseEstimator,
                 train_test_test_videos : int = 3,
                 random_state = None,
                 scaling : bool = True,
                 undersampling : bool = False,
                 labels : tuple = ("background", "supportedrear", "unsupportedrear", "grooming")):
        
        self.identity = "WrappedModel"
        self.meta = {}
        self.model = model
        self.features = X.columns
        self.labels = labels
        self.train_index, self.test_index = video_train_test_split(X, y, test_videos = train_test_test_videos, random_state = random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X.loc[self.train_index],X.loc[self.test_index],y.loc[self.train_index],y.loc[self.test_index]
        self.meta["has_DataFrame"] = True
        self.meta["fitted_on"] = None
        self.meta["prediction"] = None
        self.meta["evaluation"] = None
        self.meta["weights"] = None
        if scaling:
            num_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            self.X_train[num_features] = scaler.fit_transform(self.X_train[num_features])
            self.X_test[num_features] = scaler.transform(self.X_test[num_features])
            self.meta["scaler"] = scaler
        
        if undersampling:
            rus = RandomUnderSampler(random_state=random_state)
            self.X_train, self.y_train = rus.fit_resample(self.X_train,self.y_train)
            self.meta["undersampler"] = rus

    @classmethod
    def load(cls, input_path : str, X : pd.DataFrame = None, y : pd.DataFrame = None):

        obj = job.load(input_path)
        if X is not None and y is not None:
            obj.X_train, obj.X_test, obj.y_train, obj.y_test = X.loc[obj.train_index],X.loc[obj.test_index],y.loc[obj.train_index].values.ravel(),y.loc[obj.test_index].values.ravel()
            #columns in loaded dataframe
            obj.meta["has_DataFrame"] = True
        else:
            print("Object was instanced without DataFrame")
            obj.meta["has_DataFrame"] = False
        return obj
    
    def __str__(self):
        return f"{self.identity} for: {self.model} with {len(self.features)} input features\n train set '{self.train_index}'\n test set '{self.test_index}'\n modifications: {self.meta}"
    
    def check_if_DataFrame(self):
        if self.meta["has_DataFrame"] == False:
            raise KeyError("Object was instanced without DataFrame and cannot fit or predict.")
    
    def check_if_fitted(self):
        if self.meta["fitted_on"] == None:
            self.fit()
        
    def check_if_predicted(self):
        if self.meta["prediction"] == None:
            self.predict()
    
    def check_if_evaluated(self):
        if self.meta["evaluation"] == None:
            self.evaluate()

    def class_weights(self):
        raveled_y_train = self.y_train.values.ravel()
        unique, counts = np.unique(raveled_y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total_samples = len(raveled_y_train)
        n_classes = len(unique)
        self.class_weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}
        self.sample_weights = np.array([self.class_weights[y] for y in raveled_y_train])
        self.meta["weights"] = {"sample_weights" : self.sample_weights, "class_weights" : self.class_weights}
        return self.sample_weights

    def fit(self, **kwargs):
        self.check_if_DataFrame()
        raveled_y_train = self.y_train.values.ravel()
        self.model.fit(X = self.X_train, y = raveled_y_train, **kwargs)
        self.meta["fitted_on"] = self.train_index

    def predict(self, smooth_prediction_frames : int = None):
        self.check_if_DataFrame()
        self.y_pred_test = iter_predict(model = self.model, index = self.test_index, X = self.X_test, smooth_prediction_frames = smooth_prediction_frames)
        self.meta["prediction_test"] = {"y_pred" : self.y_pred_test, "y_true" : self.y_test,"video_id" : self.test_index, "smoothed" : False}
        if smooth_prediction_frames != None:
            self.meta["prediction"]["smoothed"] = smooth_prediction_frames
        
        self.y_pred_train = iter_predict(model = self.model, index = self.train_index, X = self.X_train, smooth_prediction_frames = smooth_prediction_frames)
        self.meta["prediction_train"] = {"y_pred" : self.y_pred_train, "y_true" : self.y_train,"video_id" : self.train_index, "smoothed" : False}
        if smooth_prediction_frames != None:
            self.meta["prediction"]["smoothed"] = smooth_prediction_frames

        return self.y_pred_test

    def evaluate(self):
        self.check_if_predicted()

        self.accuracy_train = accuracy_score(self.meta["prediction_train"]["y_true"], self.meta["prediction_train"]["y_pred"])
        self.confusion_matrix_train = confusion_matrix(self.meta["prediction_train"]["y_true"], self.meta["prediction_train"]["y_pred"], labels = self.labels)
        self.classification_report_train = classification_report(self.meta["prediction_train"]["y_true"], self.meta["prediction_train"]["y_pred"], labels = self.labels, output_dict = True)
        self.meta["evaluation_train"] = {"accuracy" : self.accuracy_train, "confusion_matrix" : self.confusion_matrix_train, "classification_report" : self.classification_report_train}

        self.accuracy_test = accuracy_score(self.meta["prediction_test"]["y_true"], self.meta["prediction_test"]["y_pred"])
        self.confusion_matrix_test = confusion_matrix(self.meta["prediction_test"]["y_true"], self.meta["prediction_test"]["y_pred"], labels = self.labels)
        self.classification_report_test = classification_report(self.meta["prediction_test"]["y_true"], self.meta["prediction_test"]["y_pred"], labels = self.labels, output_dict = True)
        self.meta["evaluation_test"] = {"accuracy" : self.accuracy_test, "confusion_matrix" : self.confusion_matrix_test, "classification_report" : self.classification_report_test}

        print("\n=== Classification Report - TRAIN SET ===")
        print(f"Train Accuracy: {(self.accuracy_train):.4f}")
        print(classification_report(self.meta["prediction_train"]["y_true"], self.meta["prediction_train"]["y_pred"], labels = self.labels))
        print(self.confusion_matrix_train)

        print("\n=== Classification Report - TEST SET ===")
        print(f"Test Accuracy: {(self.accuracy_test):.4f}")
        print("\n=== Confusion Matrix ===")
        print(self.confusion_matrix_test)
        print(classification_report(self.meta["prediction_test"]["y_true"], self.meta["prediction_test"]["y_pred"], labels = self.labels))


    def save(self, output_path : str):
        print("saving...")
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        del self.X_train, self.X_test, self.y_train, self.y_test
        job.dump(self, output_path)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
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
                 undersampling : bool = False,
                 labels : tuple = ("background", "supportedrear", "unsupportedrear", "grooming")):
        
        super().__init__(X = X,
                 y = y,
                 model = None,  # We don't have the best estimator yet
                 train_test_test_videos = train_test_test_videos,
                 random_state = random_state,
                 scaling = scaling,
                 undersampling = undersampling,
                 labels = labels)
        
        self.grid = GridSearchCV(estimator = estimator,
            param_grid = param_grid,
            scoring = scoring,     
            cv = cv,                 
            verbose = 2,
            n_jobs = n_jobs             
        )
        self.identity = "WrappedGrid"

    def fit_grid(self, **kwargs):
        raveled_y_train = self.y_train.values.ravel()
        self.grid.fit(y = raveled_y_train,X = self.X_train, **kwargs)
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
            undersampling = self.undersampling,
            labels = self.labels)
        
        return wrappedmodel
        

    
