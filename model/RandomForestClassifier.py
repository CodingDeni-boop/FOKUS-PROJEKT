
from sklearn.ensemble import RandomForestClassifier
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
from FeatureSelection import *


y = pd.read_csv("nataliia_labels.csv", index_col=["video_id","frame"])
X = pd.read_csv("features.csv", index_col=["video_id","frame"])
X = drop_non_analyzed_videos(X,y)
X, y = drop_last_frame(X,y)

# Missing Data
"""
print("X shape:", X.shape)
print("X NA:", X.isnull().sum())
print("y NA:", y.isna().sum())
"""

######################################### MISSING DATA ###########################################

# For time series data, forward fill then backward fill is most appropriate
X = X.fillna(method='ffill').fillna(method='bfill')
# If any NaN still remain (e.g., entire columns are NaN), fill with 0
X = X.fillna(0)

# Train/Test Split
X_train, X_test, y_train, y_test = video_train_test_split(
    X, y, test_videos=2)   ### takes seperate vidoes as test set

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced', 
    n_jobs=-1,
    verbose=True)

# Evaluate model
#evaluate_model(rf, X_train, y_train, X_test, y_test)

######################################## HYPERPARAMETER TUNING #################################################
"""
# n_estimators
from sklearn.metrics import f1_score

n_estimators = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
best_f1 = 0
best_n_estimators = None

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Calculate F1 score - use 'weighted' for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_n_estimators = n
    print(f"n_estimators: {n}, F1 Score: {f1:.4f}")

print(f"Best n_estimators: {best_n_estimators}, Best F1 Score: {best_f1:.4f}")
"""
# Best n_estimators: 150, Best F1 Score: 0.8932

# Feature Selection

#RecursiveFS(rf, X_train, y_train)
#RecursiveFS_CV(rf, X_train, y_train)

#UnivariateFS(rf, X_train, y_train, X_test, y_test)
