from PerformanceEvaluation import evaluate_model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

def collinearity_then_uvfs(X_train : pd.DataFrame,X_test : pd.DataFrame,y_train : pd.DataFrame,y_test : pd.DataFrame,collinearity_threshold = 0.9,uvfs_k = None, 
                           do_pretty_graphs = False):
    
    selector = SelectKBest(score_func=f_classif, k=uvfs_k)
    correlation_matrix = X_train.corr("pearson").abs()

    if do_pretty_graphs:
        correlation_matrix_for_graph = X_train.corr("pearson")
        plt.figure(figsize=(40,30))
        sns.heatmap(correlation_matrix_for_graph,vmin=-1,vmax=1)
        plt.savefig("./Eval_output/correalation_matrix_before_dropping_collinear_features.png")

    if do_pretty_graphs:
        correlation_matrix_for_graph = X_train.corr("pearson").abs()
        for i in range(0,correlation_matrix_for_graph.shape[0]):
            correlation_matrix_for_graph.iloc[i,i]=0
        max_corr = correlation_matrix_for_graph.max(axis=0)

        selector.fit(X_train, y_train)
        scores = selector.scores_
        norm_scores = scores / np.max(scores)

        feature_importance = pd.DataFrame({ 
            "column_name": X_train.columns,
            "importance": norm_scores,
            "correlation": [max_corr[col] for col in X_train.columns]})
        feature_importance=feature_importance.sort_values(by="importance",ignore_index=True,ascending=False)
        plt.figure(figsize=(100,60))
        plt.title("Normalized Feature Importance with Univariate Feature Selection; color is the maximum correlation with another feature")
        sns.barplot(data=feature_importance,y="column_name",x="importance",hue="correlation",palette="viridis") ##rocket, viridis, cubehelix
        plt.savefig("./Eval_output/Univariate_Feature_Selection_before_collinearity_drop.png")
        

    relevant_matrix = []

    for row in correlation_matrix:
        temp = correlation_matrix[row].sort_values(axis=0,ascending=False)
        temp.pop(temp.name)
        temp = temp[temp > collinearity_threshold]

        if temp.size>0:
            already_contained=False
            series = []
            series.append(temp.name)
            for item in temp.index:
                series.append(item)
            series.sort()
            for element in series:
                for series2 in relevant_matrix:
                    for element2 in series2:
                        if element2==element:
                            if len(series2)>=len(series):
                                already_contained = True
                            else:
                                relevant_matrix.remove(series2)
                                print(f"series in matrix was dropped because:{series} length {len(series)} > {series2} length {len(series2)}\n\n")
            if not already_contained:
                relevant_matrix.append(series)
                
    selector.fit(X_train, y_train)
    scores = selector.scores_
    norm_scores = scores / np.max(scores)
    feature_importance = pd.DataFrame({"column_name":pd.Series(X_train.columns),"importance":pd.Series(norm_scores)})
    for series in relevant_matrix:
        todrop = feature_importance.loc[feature_importance["column_name"].isin(series)]
        todrop = todrop.sort_values(by="importance",ascending=False,ignore_index=True)
        todrop.drop(0,inplace=True)
        X_train.drop(columns = todrop["column_name"],inplace=True)
        X_test.drop(columns = todrop["column_name"],inplace=True)
        print(X_train.shape[1]*"-")
    
    if do_pretty_graphs:
        correlation_matrix_for_graph = X_train.corr("pearson")
        plt.figure(figsize=(40,30))
        sns.heatmap(correlation_matrix_for_graph,vmin=-1,vmax=1)
        plt.savefig("./Eval_output/correalation_matrix_after_dropping_collinear_features.png")

        correlation_matrix_for_graph = X_train.corr("pearson").abs()
        for i in range(0,correlation_matrix_for_graph.shape[0]):
            correlation_matrix_for_graph.iloc[i,i]=0
        max_corr = correlation_matrix_for_graph.max(axis=0)

        selector.fit(X_train, y_train)
        scores = selector.scores_
        norm_scores = scores / np.max(scores)

        feature_importance = pd.DataFrame({ 
            "column_name": X_train.columns,
            "importance": norm_scores,
            "correlation": [max_corr[col] for col in X_train.columns]})
        feature_importance=feature_importance.sort_values(by="importance",ignore_index=True,ascending=False)
        
    if do_pretty_graphs:
        correlation_matrix_for_graph = X_train.corr("pearson").abs()
        for i in range(0,correlation_matrix_for_graph.shape[0]):
            correlation_matrix_for_graph.iloc[i,i]=0
        max_corr = correlation_matrix_for_graph.max(axis=0)

        selector.fit(X_train, y_train)
        scores = selector.scores_
        norm_scores = scores / np.max(scores)

        feature_importance = pd.DataFrame({ 
            "column_name": X_train.columns,
            "importance": norm_scores,
            "correlation": [max_corr[col] for col in X_train.columns]})
        feature_importance=feature_importance.sort_values(by="importance",ignore_index=True,ascending=False)
        plt.figure(figsize=(100,60))
        plt.title("Normalized Feature Importance with Univariate Feature Selection; color is the maximum correlation with another feature")
        sns.barplot(data=feature_importance,y="column_name",x="importance",hue="correlation",palette="viridis") ##rocket, viridis, cubehelix
        plt.savefig("./Eval_output/Univariate_Feature_Selection_after_collinearity_drop.png")

    if not uvfs_k==None:

        selector.fit(X_train, y_train)
        X_train = X_train.loc[:,selector.get_support()]
        X_test = X_test.loc[:,selector.get_support()]
    
    X_train.to_csv("./filtered_X_train.csv")
    X_test.to_csv("./filtered_X_test.csv")
    pd.DataFrame(y_train).to_csv("./filtered_y_train.csv",header=False)
    pd.DataFrame(y_test).to_csv("./filtered_y_test.csv",header=False)

    return X_train,X_test,y_train,y_test