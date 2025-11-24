import os
import shutil
import pandas as pd
import random as rd
from imblearn.under_sampling import RandomUnderSampler
import joblib
import json
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def video_train_test_split(X : pd.DataFrame,y : pd.DataFrame ,test_videos : int ,random_state=None):

    """
    Just like train test split from sklearn, but cooler. \n
    - test_videos : how many videos in the test set?

    returns X_train, X_test, y_train, y_test
    """

    rd.seed(random_state)
    X_index = X.index.get_level_values("video_id").unique()
    y_index = y.index.get_level_values("video_id").unique()
    if not (X_index.equals(y_index)):
        raise ValueError("X index name doesn't match y index name")
    index = X_index
    test_index = rd.sample(list(index),test_videos)
    train_index = index.drop(test_index)

    X_train, X_test, y_train, y_test = X.loc[train_index],X.loc[test_index],y.loc[train_index],y.loc[test_index]
    print(f"videos selected for test are: {test_index}")
    return X_train, X_test, y_train, y_test

def undersample(X_train : pd.DataFrame, y_train : pd.DataFrame,random_state=None):
    """
    resamples with RandomUnderSampler\n
    - X_train : pd.DataFrame
    - y_train : pd.DataFrame
    - random_state = None | 42 | any
    """
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X_train,y_train)
