from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
from FeatureSelection import *
from Prepare_Data import load_and_prepare_data
from SVM_Daniele import svmModel

X_train, X_test, y_train, y_test = load_and_prepare_data()

svmModel(X_train, y_train, X_test, y_test)

"""
################################ Basic SVM Model ###########################################################################

svm_basic = SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)
svm_basic.fit(X_train, y_train)

print("=== Basic SVM Model ===")
evaluate_model(svm_basic, X_train, y_train, X_test, y_test)


######################################## HYPERPARAMETER TUNING WITH FEATURE SELECTION #################################################

print("\n=== SVM with Feature Selection and Hyperparameter Tuning ===")

# Pipeline with feature selection and SVM
pipe = Pipeline([
    ("feature_selection", SelectKBest(f_classif)),
    ("svm", SVC(class_weight='balanced', probability=True, random_state=42))
])

# Hyperparameter grid
param_grid = {
    "feature_selection__k": [500, 750, 1000, 1250, 1500],
    "svm__C": [10, 50, 100, 150, 200],
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__gamma": ["scale", "auto"]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=5,
    verbose=2,
    n_jobs=-1
)

print("Starting GridSearchCV... This may take a while.")
grid_search.fit(X_train, y_train)

# Best model and parameters
best_svm = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\nBest hyperparameters: {best_params}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Evaluate best model
print("\n=== Best SVM Model Performance ===")
evaluate_model(best_svm, X_train, y_train, X_test, y_test)

# Get selected features
selected_k = best_params['feature_selection__k']
feature_selector = best_svm.named_steps['feature_selection']
selected_feature_indices = feature_selector.get_support()
selected_features = X_train.columns[selected_feature_indices].tolist()

print(f"\nNumber of features selected: {selected_k}")
print(f"Top 20 selected features by F-score:")
feature_scores = pd.DataFrame({
    'feature': X_train.columns,
    'f_score': feature_selector.scores_
}).sort_values('f_score', ascending=False)
print(feature_scores.head(20))

"""