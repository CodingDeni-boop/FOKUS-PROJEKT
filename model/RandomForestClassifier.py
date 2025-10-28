
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

start = time.time()

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
    f1 = f1_score(y_test, y_pred, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_n_estimators = n
    print(f"n_estimators: {n}, F1 Score: {f1:.4f}")

print(f"Best n_estimators: {best_n_estimators}, Best F1 Score: {best_f1:.4f}")

# Best n_estimators: 150, Best F1 Score: 0.8932
rf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    verbose=True)
"""

########################################### Data loading ##########################################################
"""
# Apply UVFS
X_train_sel, X_test_sel, selected_features, feature_scores_df = apply_uvfs(X_train, X_test, y_train, k_best=100)
"""

# UVFS + Collinearity
X_train, X_test, y_train, y_test = preprocess_data()
X_train_sel, X_test_sel, y_train, y_test = collinearity_then_uvfs(X_train, X_test, y_train, y_test, collinearity_threshold = 0.95)

############################################# Basic Model ####################################

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, n_estimators=150,max_depth=10,min_samples_split=5,min_samples_leaf=2,max_features='sqrt')
rf.fit(X_train_sel, y_train)

evaluate_model(rf, X_train_sel, y_train, X_test_sel, y_test)

############################################ Hyperparamter Tuning ##############
"""
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
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

grid_search.fit(X_train_sel, y_train)

best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

evaluate_model(best_rf, X_train_sel, y_train, X_test_sel, y_test)
"""

########################################### Model Training #############################################################
"""
# Train Random Forest on the selected features
rf.fit(X_train_sel, y_train)

evaluate_model(rf, X_train_sel, y_train, X_test_sel, y_test)
"""
end = time.time()
print("Time elapsed:", end - start)

