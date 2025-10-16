import os
import shutil
import pandas as pd
import random as rd

def video_train_test_split(X : pd.DataFrame,y : pd.DataFrame ,test_videos : int ,random_state=None):

    rd.seed(random_state)

    print(rd.getstate())

    X_train, X_test, y_train, y_test = 1,2,3,4

    return X_train, X_test, y_train, y_test 