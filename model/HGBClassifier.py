# Using sklearn's HistGradientBoostingClassifier for optimal performance and compatibility
from sklearn.ensemble import HistGradientBoostingClassifier

USE_HIST_GB = True


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
from DataPreprocessing import preprocess_data
import time
from DataLoading import load_data
import matplotlib.pyplot as plt
import json
from save_and_load_as_pkl import save_model_as_pkl

start = time.time()

################################ فنیData loading ###############################################################

X_train, X_test, y_train, y_test = preprocess_data(features_file="processed_features.csv",
                                                   labels_file="processed_labels.csv")

# Calculate class weights for multi-class imbalanced data
unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"Class distribution in training: {class_counts}")

# For multi-class, calculate sample weights
total_samples = len(y_train)
n_classes = len(unique)
class_weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}
sample_weights = np.array([class_weights[y] for y in y_train])
print(f"Class weights: {class_weights}")

############################################# Tuned Model ##############################################################
"""
print("Using Histogram-Based Gradient Boosting Classifier")
model = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=100,  # equivalent to n_estimators
    max_depth=6,
    learning_rate=0.1,
    max_bins=255,  # default, can increase for more precision
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=False,
    verbose=0
)

model.fit(X_train, y_train, sample_weight=sample_weights)

print("With smoothing")
evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=20)

print("Without smoothing")
evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=0)
"""
############################################ Hyperparameter Tuning #####################################################

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_iter': [50, 100, 150, 200],
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_samples_leaf': [10, 20, 30, 40],
    'l2_regularization': [0.0, 0.1, 0.5, 1.0],
    'max_bins': [255]  # can try [128, 255] if you want
}

base_model = HistGradientBoostingClassifier(
    random_state=42,
    early_stopping=False,
    verbose=0
)

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=2,
    verbose=2
)

grid_search.fit(X_train, y_train, sample_weight=sample_weights)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Save best parameters as JSON
best_params = grid_search.best_params_
with open('best_parameters_hgb.json', 'w') as f:
    json.dump(best_params, f, indent=4)

evaluate_model(best_model, X_train, y_train, X_test, y_test)


############################################### Feature Importance #####################################################

# Extract feature importances
importances = best_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(100))

# Save feature importances to CSV
feature_importance_df.to_csv('feature_importances_model.csv', index=False)

# Plot top 50 feature importances
top_n_plot = 50
top_features_plot = feature_importance_df.head(top_n_plot)
plt.figure(figsize=(10, 12))
plt.barh(range(top_n_plot), top_features_plot['Importance'], align='center')
plt.yticks(range(top_n_plot), top_features_plot['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
model_name = "LightGBM" if USE_HIST_GB else "Gradient Boosting"
plt.title(f'Top {top_n_plot} {model_name} Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importances_model.png', dpi=300, bbox_inches='tight')
plt.close()

################################################ Grid Search for Number of Features ####################################

best_n_features = 750  # You can adjust this based on your RF results

################################################# New Model with selected features ####################################

n = best_n_features
top_features = feature_importance_df['Feature'][:n].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Load the parameters
with open('best_parameters_hgb.json', 'r') as f:
    best_parameters = json.load(f)

print("Using Histogram-Based Gradient Boosting Classifier")
model_selected = HistGradientBoostingClassifier(**best_parameters)

model_selected.fit(X_train_selected, y_train, sample_weight=sample_weights)

print("=" * 80)
model_name = "LightGBM" if USE_HIST_GB else "Gradient Boosting"
print(f"New {model_name} Classifier with selected features ({n}):")
print("=" * 80)

print("With smoothing")
evaluate_model(model_selected, X_train_selected, y_train, X_test_selected, y_test, min_frames=20)
print("Without smoothing")
evaluate_model(model_selected, X_train_selected, y_train, X_test_selected, y_test, min_frames=0)

end = time.time()
print("Time elapsed:", end - start)

# Save model
save_model_as_pkl(name="gradient_boost", folder="GB_model", model=best_model, columns=X_train.columns, random_state=42)