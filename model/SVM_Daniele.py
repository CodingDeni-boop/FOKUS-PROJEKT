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
from save_and_load_as_pkl import save_model_as_pkl


start=time.time()

X_train, X_test, y_train, y_test = preprocess_data(
    features_file="processed_features.csv",
    labels_file="processed_labels.csv",
    random_state=42,
    test_videos = 1
)

#X_train,y_train = undersample(X_train,y_train)

#print("!undersampled!")

# X_train, X_test, y_train, y_test = collinearity_then_uvfs(X_train, X_test, y_train, y_test,collinearity_threshold=0.95)

print(X_train, X_test, y_train, y_test)

svm = SVC(probability=True,C=0.05,coef0=1,degree=3,kernel="poly")
svm.fit(X_train, y_train)

save_model_as_pkl(name = "poly_3_not_undersampled", model=svm,columns=X_train.columns,random_state=42)

end=time.time()
print("time elapsed:", f"{int((end-start)//3600)}h {int(((end-start)%3600)//60)}m {int((end-start)%60)}s")

pipe = Pipeline({
        ("filter", SelectKBest()),
        ("SVM", SVC(class_weight="balanced", probability=True))
    })

hyperparameters={
    "filter__k" : [500, 1000, 1250, 1500, 1750],
    "SVM__C" : [10,100,125,150,175],
    "SVM__kernel" : ["linear", "poly", "rbf", "sigmoid"]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=hyperparameters,
    scoring="f1",     
    cv=5,                 
    verbose=2,
    n_jobs=-1             
)
grid.fit(y=y,X=X)
bestFit = grid.best_estimator_
bestHyperparameters = grid.best_params_
print(f"The best hyperparameters selected were:   {bestHyperparameters}")
