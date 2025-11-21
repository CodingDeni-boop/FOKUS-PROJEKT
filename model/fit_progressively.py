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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from DataPreprocessing import preprocess_data
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from FeatureSelection import collinearity_then_uvfs
from sklearn.linear_model import LinearRegression as SomeModel
from sklearn.svm import SVC as SVM
from sklearn.metrics import f1_score

def fit_progressively(model : SomeModel,X : pd.DataFrame, y : pd.DataFrame):

    f1s = []

    for i in range(2, y["video_id"].unique().size):
        
        X_train, X_test, y_train, y_test = video_train_test_split(X,y,test_videos = np.ceil(i*0.2))

        model.fit(X_train,y_train)
        f1 = f1_score(y_test, model.predict(X_test))
        f1s.append(f1)
        print(f"f1 for {i} videos was {f1}")
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(range(2, y["video_id"].unique().size),f1s)
    plt.savefig("./Eval_output/f1_score_vs_videos.png")
    plt.show()


fit_progressively(SVM(C= 0.05, coef0=1, degree=3, kernel="poly"), pd.read_csv("./processed_features_lite.csv"),pd.read_csv("./processed_labels_lite.csv"))