
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
from DataPreprocessing import preprocess_data
from FeatureSelection import apply_uvfs, apply_pca
import time
from FeatureSelection import collinearity_then_uvfs
from DataLoading import load_data

start = time.time()

# Best n_estimators: 150, Best F1 Score: 0.8932

########################################### Data loading ###############################################################
"""
# Apply UVFS
X_train_sel, X_test_sel, selected_features, feature_scores_df = apply_uvfs(X_train, X_test, y_train, k_best=100)
"""

load_data()
X_train, X_test, y_train, y_test = preprocess_data()

# UVFS + Collinearity
#X_train_sel, X_test_sel, y_train, y_test = collinearity_then_uvfs(X_train, X_test, y_train, y_test, collinearity_threshold = 0.95)

############################################# Basic Model ##############################################################

#rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, n_estimators=150,max_depth=10,min_samples_split=5,min_samples_leaf=2,max_features='sqrt')
#rf.fit(X_train, y_train)

#evaluate_model(rf, X_train, y_train, X_test, y_test)

############################################ Hyperparameter Tuning #####################################################

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [125, 200, 300],
    'max_depth': [5, 10, 20, 30],
    'min_samples_split': [10, 20,30],
    'min_samples_leaf': [ 4, 8, 16],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

evaluate_model(best_rf, X_train, y_train, X_test, y_test)


end = time.time()
print("Time elapsed:", end - start)

