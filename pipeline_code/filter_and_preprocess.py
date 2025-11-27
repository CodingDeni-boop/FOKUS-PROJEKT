import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def reduce_bits(X : pd.DataFrame):
    float64_cols = list(X.select_dtypes(include='float64'))
    X[float64_cols] = X[float64_cols].astype('float32')
    return X

def scale(X_train, X_test):
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train, X_test

