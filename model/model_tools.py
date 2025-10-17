import os
import shutil
import pandas as pd
import random as rd

def drop_non_analyzed_videos(X : pd.DataFrame,y : pd.DataFrame):

    X_index = X.index.get_level_values("video_id").unique()
    y_index = y.index.get_level_values("video_id").unique()
    return X.loc[y_index]

def drop_last_frame(X : pd.DataFrame,y : pd.DataFrame):
    
    X_index = X.index.get_level_values("video_id").unique()
    y_index = y.index.get_level_values("video_id").unique()
    if not (X_index.equals(y_index)):
        raise ValueError("X index name doesn't match y index name")
    index = X_index
    while X.shape[0]!=y.shape[0]:
        for video_name in index:
            if y.loc[video_name].shape[0] == X.loc[video_name].shape[0]:
                continue

            elif y.loc[video_name].shape[0] > X.loc[video_name].shape[0]:
                difference = y.loc[video_name].shape[0] - X.loc[video_name].shape[0]
                y = y.drop((video_name, y.loc[video_name].index[-1]))
                print(f"{video_name} has {difference} too many frames in y: dropped 1")

            elif y.loc[video_name].shape[0] < X.loc[video_name].shape[0]:
                difference = X.loc[video_name].shape[0] - y.loc[video_name].shape[0]
                X = X.drop((video_name, y.loc[video_name].index[-1]))
                print(f"{video_name} has {difference} too many frames in X: dropped 1")

    return X, y


def video_train_test_split(X : pd.DataFrame,y : pd.DataFrame ,test_videos : int ,random_state=None):

    """
    Just like train test split from sklearn, but cooler. \n
    - test_videos : how many videos in the test set?
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

    return X_train, X_test, y_train, y_test