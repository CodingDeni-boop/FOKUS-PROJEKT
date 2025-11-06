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
from save_and_load_as_pkl import load_model_from_pkl



X_train, X_test, y_train, y_test, model = load_model_from_pkl(name = "poly_3_not_undersampled",random_state=42,
                                                              features_path = "processed_features.csv",
                                                              labels_path="processed_labels.csv",
                                                              folder = "SVM_(hyper)parameters/6hr_SVM")

print(X_train)

y_test = pd.Series(y_test,name = "label")

print("predicting...")
y_pred = pd.Series(model.predict(X_test),name="label")

print("predicting probabilities...")
y_proba = pd.DataFrame(model.predict_proba(X_test))

print("saving data...")
y_pred.to_csv("./prediction.csv",index=0)
y_test.to_csv("./test.csv",index=0)
y_proba.to_csv("./proba.csv")

evaluate_model(model,X_train,y_train,X_test,y_test)

