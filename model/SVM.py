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
import time

start = time.time()

# Load data
y = pd.read_csv("nataliia_labels.csv", index_col=["video_id", "frame"])
X = pd.read_csv("features.csv", index_col=["video_id", "frame"])
X = drop_non_analyzed_videos(X, y)
X, y = drop_last_frame(X, y)

######################################### MISSING DATA ###########################################

na_percentage = X.isna().mean()
columns_to_keep = na_percentage[na_percentage <= 0.1].index
columns_dropped = na_percentage[na_percentage > 0.1].index

print(f"Dropped {len(columns_dropped)} columns with >10% missing values:")
print(columns_dropped.tolist())
X = X[columns_to_keep]

valid_mask = X.notna().all(axis=1)
valid_X = X[valid_mask]
valid_y = y[valid_mask]

####################################### Train/Test Split ##########################################

X_train, X_test, y_train, y_test = video_train_test_split(
    valid_X, valid_y, test_videos=2)

# Convert labels to numpy arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

#######################  SCALING (CRITICAL FOR SVM!) ####################
from sklearn.preprocessing import StandardScaler

num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

sc = StandardScaler()
X_train[num_features] = sc.fit_transform(X_train[num_features])
X_test[num_features] = sc.transform(X_test[num_features])

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

