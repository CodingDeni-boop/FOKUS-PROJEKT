from PerformanceEvaluation import evaluate_model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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

    print("Recursive Feature Elimination: ")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)


# Recursive Feature Elimination with Cross Validation
def RecursiveFS_CV(rf, X_train, y_train, X_test, y_test):
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

    print("Recursive Feature Elimination with Cross Validation: ")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)

    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'])
    plt.xlabel('Number of Features')
    plt.ylabel('CV Score (F1 Weighted)')
    plt.title('RFECV Feature Selection')
    plt.show()

"""
# Univariate Feature Selection
def UnivariateFS(rf, X_train, y_train, X_test, y_test):
    from sklearn.feature_selection import SelectKBest, f_classif
    # Select top K features
    k = 20
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

    print("Univariate Feature Selection: ")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)
"""

def UnivariateFS(apply_pca=False, n_components=0.95, k=20):
    """
    Perform Univariate Feature Selection after data preprocessing.
    Returns transformed datasets and feature information for model training.

    Parameters
    ----------
    apply_pca : bool, default=False
        Whether to apply PCA in preprocessing. PCA should generally be False for feature-level interpretability.
    n_components : float or int, default=0.95
        Number of PCA components or explained variance ratio.
    k : int, default=20
        Number of top features to select.

    Returns
    -------
    X_train_selected : np.ndarray
        Transformed training data with selected features.
    X_test_selected : np.ndarray
        Transformed test data with selected features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    selected_features : list
        Names of selected features.
    scores_df : pd.DataFrame
        F-scores and p-values for all features.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from DataPreprocessing import preprocess_data
    import pandas as pd

    print("=" * 80)
    print("UNIVARIATE FEATURE SELECTION")
    print("=" * 80)

    # Step 1: Preprocess data
    X_train, X_test, y_train, y_test, pca, original_features  = preprocess_data()

    # Step 3: Perform univariate feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    # Step 4: Collect results
    selected_features = X_train.columns[selector.get_support()].tolist()

    scores_df = pd.DataFrame({
        'Feature': X_train.columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    }).sort_values('F_Score', ascending=False)

    print(f"\nSelected top {k} features:\n{selected_features}")
    print("\nTop 10 features by F-score:")
    print(scores_df.head(10))

    # Step 5: Transform datasets
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print("\nUnivariate feature selection complete.")

    return X_train_selected, X_test_selected, y_train, y_test, selected_features, scores_df



