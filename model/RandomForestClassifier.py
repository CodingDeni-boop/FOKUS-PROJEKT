
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

rf = RandomForestClassifier()
y = pd.read_csv("nataliia_labels.csv", index_col=["video_id","frame"])
X = pd.read_csv("features.csv", index_col=["video_id","frame"])
X = drop_non_analyzed_videos(X,y)
X, y = drop_last_frame(X,y)

# Missing Data
print("X shape:", X.shape)
print("X NA:", X.isnull().sum())
print("y NA:", y.isna().sum())

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
from sklearn.metrics import recall_score

n_estimators = [100, 150, 200, 250, 300, 400, 500]
best_recall = 0
best_n_estimators = None

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Calculate recall - use 'weighted' for multi-class, or specify average='macro'/'micro'
    recall = recall_score(y_test, y_pred, average='weighted')
    if recall > best_recall:
        best_recall = recall
        best_n_estimators = n
    print(f"n_estimators: {n}, Recall: {recall:.4f}")

print(f"Best n_estimators: {best_n_estimators}, Best Recall: {best_recall:.4f}")
"""

########################################### FEATURE SELECTION #########################################################

# Recursive Feature Elimination
def RecursiveFS(rf, X_train, y_train, X_test, y_test):
    from sklearn.feature_selection import RFE

    # Select top N features
    rfe = RFE(estimator=rf, n_features_to_select=20, step=5)
    rfe.fit(X_train, y_train)

    # Get selected features
    selected_features = X_train.columns[rfe.support_].tolist()
    print(f"Selected features: {selected_features}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print("Recursive Feature Elimination:/n")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)


# Recursive Feature Elimination with Cross Validation
def RecursiveFS_CV(rf, X_train, y_train):
    from sklearn.feature_selection import RFECV
    import matplotlib.pyplot as plt

    rfecv = RFECV(
        estimator=rf,
        step=5,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")

    # Get selected features
    selected_features = X_train.columns[rfecv.support_].tolist()
    print(f"Selected features: {selected_features}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print("Recursive Feature Elimination with Cross Validation:/n")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)

    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'])
    plt.xlabel('Number of Features')
    plt.ylabel('CV Score (Recall)')
    plt.title('RFECV Feature Selection')
    plt.show()

#RecursiveFS_CV(rf, X_train, y_train)

# Univariate Feature Selection
def UnivariateFS(rf, X_train, y_train, X_test, y_test):
    from sklearn.feature_selection import SelectKBest, f_classif
    # Select top K features
    k = 10
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    # Get selected features
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features}")

    # Get scores
    scores_df = pd.DataFrame({
        'feature': X_train.columns,
        'f_score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('f_score', ascending=False)

    print(f"Top 10 features by F-score:")
    print(scores_df.head(10))

    # Transform
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print("Univariate Feature Selection:/n")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)

UnivariateFS(rf, X_train, y_train, X_test, y_test)
