#from labels.deepethogram_vid_import import all_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
import time
from sklearn.linear_model import LogisticRegression
from Prepare_Data import load_and_prepare_data
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from Data_Preprocessing import preprocess_data


start=time.time()

X_train, X_test, y_train, y_test, pca = preprocess_data(
    features_file="features.csv",
    labels_file="nataliia_labels.csv",
    apply_pca=True,
    n_components=0.95
)

print(X_train.shape, X_test.shape)