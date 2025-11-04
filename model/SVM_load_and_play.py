from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from model_tools import undersample
from PerformanceEvaluation import evaluate_model
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from DataPreprocessing import preprocess_data
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from FeatureSelection import collinearity_then_uvfs
from DataLoading import load_data
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import json
import joblib

name = "onlypoly_lite"

print("loading model...")
model : SVC = joblib.load("./SVM_(hyper)parameters/"+name+"_model.pkl")

print("loading dataset...")
columns = joblib.load("./SVM_(hyper)parameters/"+name+"_columns.pkl")

X_train, X_test, y_train, y_test = preprocess_data(
    features_file="processed_features_lite.csv",
    labels_file="processed_labels_lite.csv",
    random_state=42
)

X_train = X_train[columns]
X_test = X_test[columns]

y_pred = pd.Series(model.predict(X_test))
y_pred.to_csv("./prediction.csv")


