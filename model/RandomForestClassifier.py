
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
#from FeatureSelection import UnivariateFS, RecursiveFS_CV
from Data_Preprocessing import preprocess_data

################################## Load Data ###########################################################################

X_train, X_test, y_train, y_test, pca , original_features = preprocess_data(
    features_file="features.csv",
    labels_file="nataliia_labels.csv",
    apply_pca=True,
    n_components=0.95
)

####################################### Basic Model ####################################################################

rf = RandomForestClassifier(
    n_estimators=150,
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

def UnivariateFS(X_train, y_train, X_test, y_test, k=20):
    """
    Perform Univariate Feature Selection (ANOVA F-test).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (preprocessed, not PCA-transformed)
    y_train : pd.Series or np.ndarray
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series or np.ndarray
        Test labels
    k : int, default=20
        Number of top features to select

    Returns
    -------
    X_train_selected : np.ndarray
        Transformed training data with selected features
    X_test_selected : np.ndarray
        Transformed test data with selected features
    selected_features : list
        Names of selected features
    scores_df : pd.DataFrame
        F-scores and p-values for all features
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    import pandas as pd

    print("=" * 80)
    print(f"UNIVARIATE FEATURE SELECTION (Top {k} Features)")
    print("=" * 80)

    # Initialize selector
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    # Get selected features
    selected_features = X_train.columns[selector.get_support()].tolist()

    # Create scores DataFrame
    scores_df = pd.DataFrame({
        'Feature': X_train.columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    }).sort_values('F_Score', ascending=False)

    print(f"\nSelected features ({len(selected_features)}): {selected_features}")
    print("\nTop 10 features by F-score:")
    print(scores_df.head(10))

    # Transform datasets
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print("\nUnivariate feature selection complete.")

    return X_train_selected, X_test_selected, selected_features, scores_df


#RecursiveFS(rf, X_train, y_train)
#RecursiveFS_CV(rf, X_train, y_train)

# Run feature selection
X_train_sel, X_test_sel, selected_features, scores_df = UnivariateFS(X_train, X_test, y_train, y_test, k=30)

# Train Random Forest on the selected features
from sklearn.ensemble import RandomForestClassifier

rf.fit(X_train_sel, y_train)

evaluate_model(rf, X_train_sel, y_train, X_test_sel, y_test)

