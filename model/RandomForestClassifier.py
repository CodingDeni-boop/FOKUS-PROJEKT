
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
import matplotlib.pyplot as plt
import json

start = time.time()


########################################### Data loading ###############################################################

X_train, X_test, y_train, y_test = preprocess_data()

############################################# Tuned Model ##############################################################

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, n_estimators=200,max_depth=10,min_samples_split=20,min_samples_leaf=8,max_features='log2')
rf.fit(X_train, y_train)

evaluate_model(rf, X_train, y_train, X_test, y_test)

############################################ Hyperparameter Tuning #####################################################

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [125, 200],
    'max_depth': [5, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [ 8, 16],
    'max_features': ['log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Save best parameters as JSON and plot
best_params = grid_search.best_params_
with open('best_parameters.json', 'w') as f:
    json.dump(best_params, f, indent=4)


evaluate_model(best_rf, X_train, y_train, X_test, y_test)

############################################### Feature Importance ##########################################

# Extract feature importances
importances = best_rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(100))

# Save feature importances to CSV
feature_importance_df.to_csv('feature_importances.csv', index=False)

# Plot top 50 feature importances
top_n_plot = 50
top_features_plot = feature_importance_df.head(top_n_plot)
plt.figure(figsize=(10, 12))
plt.barh(range(top_n_plot), top_features_plot['Importance'], align='center')
plt.yticks(range(top_n_plot), top_features_plot['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title(f'Top {top_n_plot} Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.close()

# Select top N features
n = 1000
top_features = feature_importance_df['Feature'][:n].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

################################################# New RF with selected features ########################################
"""
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

print("=" * 80)
print(f"New Random Forest Classifier with selected features ({n}):")
print("=" * 80)

evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)


end = time.time()
print("Time elapsed:", end - start)
"""
############################################# Balanced Random Forest Classifier ########################################
"""
brf = BalancedRandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=8,
    min_samples_split=20,
    max_samples=0.5,
    max_features='log2',
    bootstrap=True,
    max_depth=10,
    sampling_strategy='all',  # Balance classes by undersampling
    replacement=True,
    n_jobs=-1,
    random_state=42
)

brf.fit(X_train, y_train)

print("=" * 80)
print("Balanced Random Forest Classifier:")
print("=" * 80)

evaluate_model(brf, X_train, y_train, X_test, y_test)
"""
