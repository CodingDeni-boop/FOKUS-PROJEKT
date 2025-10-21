from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
import time

start = time.time()

# Load data
y = pd.read_csv("nataliia_labels.csv", index_col=["video_id", "frame"])
X = pd.read_csv("features.csv", index_col=["video_id", "frame"])
X = drop_non_analyzed_videos(X, y)
X, y = drop_last_frame(X, y)

print("X shape", X.shape)

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

#######################  SCALING  ####################
from sklearn.preprocessing import StandardScaler

num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

sc = StandardScaler()
X_train[num_features] = sc.fit_transform(X_train[num_features])
X_test[num_features] = sc.transform(X_test[num_features])

################################ Feature Selection ###################################
from sklearn.feature_selection import VarianceThreshold

# Remove features with very low variance
selector = VarianceThreshold(threshold=0.01)
X_train_transformed = selector.fit_transform(X_train)
X_test_transformed = selector.transform(X_test)

# Get the selected feature names
selected_features = X_train.columns[selector.get_support()]

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_transformed, columns=selected_features, index=X_train.index)
X_test = pd.DataFrame(X_test_transformed, columns=selected_features, index=X_test.index)

# Remove redundant features
correlation_matrix = X_train.corr().abs()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

print(f"Final feature count: {X_train.shape[1]}")

################################ Basic SVM Model ###########################################################################

print("\n=== Training Basic SVM (Linear Kernel) ===")
svm_linear = SVC(kernel='linear', random_state=42, class_weight='balanced')
svm_linear.fit(X_train, y_train)

evaluate_model(svm_linear, X_train, y_train, X_test, y_test)

################################ SVM with RBF Kernel ###########################################################################

print("\n=== Training SVM (RBF Kernel) ===")
svm_rbf = SVC(kernel='rbf', random_state=42, class_weight='balanced')
svm_rbf.fit(X_train, y_train)

evaluate_model(svm_rbf, X_train, y_train, X_test, y_test)

################################ Hyperparameter Tuning ###########################################################################

print("\n=== Hyperparameter Tuning with GridSearchCV ===")

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    SVC(random_state=42, class_weight='balanced'),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_svm = grid_search.best_estimator_
print("\n=== Best SVM Model Performance ===")
evaluate_model(best_svm, X_train, y_train, X_test, y_test)