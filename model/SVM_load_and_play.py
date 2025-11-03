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

print("loading model...")
model : SVC = joblib.load("./SVM_(hyper)parameters/lite_best_parameters.pkl")

print("loading dataset...")
X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.Series
y_test: pd.Series
X_train, X_test, y_train, y_test = joblib.load("./SVM_(hyper)parameters/dataset.pkl")[:]

print("!loaded!")

evaluate_model(model,X_train,y_train,X_test, y_test)
