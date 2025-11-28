import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest

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

def collinearity_then_uvfs(X_train : pd.DataFrame,
                           X_test : pd.DataFrame,
                           y_train : pd.DataFrame, 
                           feature_importance_dataframe : pd.DataFrame, 
                           collinearity_threshold = 0.9):

    correlation_matrix = X_train.corr()
    relevant_matrix = []

    for row in correlation_matrix:
        temp = correlation_matrix[row].sort_values(axis=0,ascending=False)
        temp.pop(temp.name)
        temp = temp[temp > collinearity_threshold]

        if temp.size>0:
            already_contained=False
            series = []
            series.append(temp.name)
            for item in temp.index:
                series.append(item)
            series.sort()
            for element in series:
                for series2 in relevant_matrix:
                    for element2 in series2:
                        if element2==element:
                            if len(series2)>=len(series):
                                already_contained = True
                            else:
                                relevant_matrix.remove(series2)
            if not already_contained:
                relevant_matrix.append(series)
    
    for series in relevant_matrix:
        todrop = feature_importance_dataframe.loc[feature_importance_dataframe["feature"].isin(series)]
        todrop = todrop.sort_values(by="score",ascending=False,ignore_index=True)
        todrop.drop(0,inplace=True)
        X_train.drop(columns = todrop["feature"],inplace=True)
        X_test.drop(columns = todrop["feature"],inplace=True)
    
    return X_train, X_test