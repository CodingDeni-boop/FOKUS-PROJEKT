
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
from sklearn.feature_selection import VarianceThreshold
from imblearn.ensemble import BalancedRandomForestClassifier

start = time.time()


########################################### Data loading ###############################################################


X_train, X_test, y_train, y_test = preprocess_data()

# Remove low-variance features
variance_selector = VarianceThreshold(threshold=0.001)
X_train = variance_selector.fit_transform(X_train)
X_test = variance_selector.transform(X_test)

print(f"After variance filtering: {X_train.shape[1]} features")



############################################# Tuned Model ##############################################################

rf = RandomForestClassifier(random_state=2, class_weight='balanced', n_jobs=-1, n_estimators=200,max_depth=10,min_samples_split=20,min_samples_leaf=8,max_features='log2')
rf.fit(X_train, y_train)

evaluate_model(rf, X_train, y_train, X_test, y_test)

############################################ Hyperparameter Tuning #####################################################
"""
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
"""
############################################### Feature Importance ##########################################

# Extract feature importances
importances = rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Select top N features
top_features = feature_importance_df['Feature'][:500].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

################################################# New RF with selected features ########################################

rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=8,
    max_features='log2')
rf.fit(X_train_selected, y_train)

evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)


end = time.time()
print("Time elapsed:", end - start)

############################################# Balanced Random Forest Classifier ########################################

brf = BalancedRandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=8,
    min_samples_split=20,
    max_depth=10,
    sampling_strategy='all',  # Balance classes by undersampling
    replacement=True,
    n_jobs=-1,
    random_state=42
)

brf.fit(X_train, y_train)

evaluate_model(brf, X_train, y_train, X_test, y_test)

