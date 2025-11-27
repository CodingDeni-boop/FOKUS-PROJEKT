import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def reduce_bits(X : pd.DataFrame):
    float64_cols = list(X.select_dtypes(include='float64'))
    X[float64_cols] = X[float64_cols].astype('float32')
    return X

class Filter:
    def __init__(self, filter_object: VarianceThreshold):
        self.filter_object = filter_object
    
    def fit(self, X_fit: pd.DataFrame):
        self.X_fit = X_fit
        self.feature_names = X_fit.columns
        self.filter_object.fit(X_fit)
    
    def get_variance(self):
        variances = self.filter_object.variances_
        normalized_variances = variances / variances.max()

        self.importance = pd.DataFrame({
            "feature": self.feature_names,
            "variance": normalized_variances,
        })
        self.importance.sort_values(by = "variance", ascending = False, inplace = True)
    
    def graph_variance(self,path : str):
        self.get_variance()
        if not os.path.isfile(path):
            plt.figure(figsize=(10,20))
            sns.barplot(y = self.importance["feature"], x = self.importance["variance"])
            plt.subplots_adjust(left=0.5)
            plt.savefig(path)
            plt.close()
    
    def transform(self, X: pd.DataFrame):
        to_drop = self.feature_names.drop(self.filter_object.get_feature_names_out())
        print(f"These features were dropped: {to_drop} with variance threshold {self.filter_object.get_params()['threshold']}")
        for name in to_drop:
            for column in X.columns:
                if name in column:
                    X.drop(columns=column)
                    print(f"dropped {column} cause {name}")
        print(X)
        return X

