from PerformanceEvaluation import evaluate_model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

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

def apply_uvfs(X_train, X_test, y_train, k_best=30):
    """
    Apply Univariate Feature Selection (UVFS) to select top k features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : array-like
        Training labels
    k_best : int, default=30
        Number of top features to select

    Returns
    -------
    X_train_selected : pd.DataFrame
        Training data with selected features
    X_test_selected : pd.DataFrame
        Test data with selected features
    selected_feature_names : list
        Names of selected features
    feature_scores_df : pd.DataFrame
        DataFrame with feature scores and p-values
    """
    print(f"\nApplying Univariate Feature Selection (top {k_best} features)...")

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train[num_features], y_train)
    X_test_selected = selector.transform(X_test[num_features])

    selected_feature_mask = selector.get_support()
    selected_feature_names = X_train[num_features].columns[selected_feature_mask].tolist()

    # Create DataFrame with feature scores
    feature_scores_df = pd.DataFrame({
        'Feature': X_train[num_features].columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    }).sort_values(by='F_Score', ascending=False)

    print("\n    Top Selected Features by UVFS:")
    print(feature_scores_df.head(k_best).to_string(index=False))

    # Convert to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, index=X_train.index, columns=selected_feature_names)
    X_test_selected = pd.DataFrame(X_test_selected, index=X_test.index, columns=selected_feature_names)

    return X_train_selected, X_test_selected, selected_feature_names, feature_scores_df


def apply_pca(X_train, X_test, n_components=0.95):
    """
    Apply PCA for dimensionality reduction.

    Parameters
    ----------
    X_train : pd.DataFrame or array-like
        Training features
    X_test : pd.DataFrame or array-like
        Test features
    n_components : float or int, default=0.95
        Number of components or explained variance ratio

    Returns
    -------
    X_train_pca : pd.DataFrame
        Training data after PCA transformation
    X_test_pca : pd.DataFrame
        Test data after PCA transformation
    pca : PCA object
        Fitted PCA transformer
    """
    print(f"\nApplying PCA (n_components={n_components})...")

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca_array = pca.fit_transform(X_train)
    X_test_pca_array = pca.transform(X_test)

    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"   PCA reduced dimensions: {X_train_pca_array.shape[1]} components")
    print(f"   Total explained variance: {explained_var:.2f}%")

    # Convert to DataFrame
    X_train_pca = pd.DataFrame(
        X_train_pca_array,
        index=X_train.index,
        columns=[f"PC{i+1}" for i in range(X_train_pca_array.shape[1])]
    )
    X_test_pca = pd.DataFrame(
        X_test_pca_array,
        index=X_test.index,
        columns=[f"PC{i+1}" for i in range(X_test_pca_array.shape[1])]
    )

    return X_train_pca, X_test_pca, pca


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
    from DataPreprocessing import preprocess_data

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



