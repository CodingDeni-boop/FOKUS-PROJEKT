import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model_tools import video_train_test_split, drop_non_analyzed_videos, drop_last_frame

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from model_tools import video_train_test_split, drop_non_analyzed_videos, drop_last_frame


def preprocess_data(
        features_file="features.csv",
        labels_file="nataliia_labels.csv",
        missing_threshold=0.1,
        test_videos=2,
        apply_uvfs=False,
        k_best=30,
        apply_pca=False,
        n_components=0.95  # keep 95% variance by default
):
    """
    Load data, handle missing values, split, scale, optionally apply
    Univariate Feature Selection (UVFS) and PCA.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays or DataFrames
    pca : PCA object or None
    original_features : list of UVFS-selected feature names (before PCA)
    """

    print("=" * 70)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 70)

    # ------------------------------
    # 1. Load data
    # ------------------------------
    print("\n1. Loading data...")
    y = pd.read_csv(labels_file, index_col=["video_id", "frame"])
    X = pd.read_csv(features_file, index_col=["video_id", "frame"])
    X = drop_non_analyzed_videos(X, y)
    X, y = drop_last_frame(X, y)
    print(f"   Initial X shape: {X.shape}")

    # ------------------------------
    # 2. Handle missing values
    # ------------------------------
    print(f"\n2. Handling missing data (threshold={missing_threshold * 100:.0f}%)...")
    na_percentage = X.isna().mean()
    columns_to_keep = na_percentage[na_percentage <= missing_threshold].index
    X = X[columns_to_keep]
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"   Final X shape after cleaning: {X.shape}")

    # ------------------------------
    # 3. Train/Test split
    # ------------------------------
    print(f"\n3. Splitting data (test_videos={test_videos})...")
    X_train, X_test, y_train, y_test = video_train_test_split(X, y, test_videos=test_videos, random_state=42)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ------------------------------
    # 4. Scaling
    # ------------------------------
    print("\n4. Scaling numeric features...")
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    print("   Scaling complete.")

    # ------------------------------
    # 5. Univariate Feature Selection (UVFS)
    # ------------------------------
    if apply_uvfs:
        print(f"\n5. Applying Univariate Feature Selection (top {k_best} features)...")
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

        # Replace numeric columns with selected features
        X_train = pd.DataFrame(X_train_selected, index=X_train.index, columns=selected_feature_names)
        X_test = pd.DataFrame(X_test_selected, index=X_test.index, columns=selected_feature_names)

        original_features = selected_feature_names
    else:
        original_features = num_features
        print("\n5. Skipping Univariate Feature Selection (UVFS).")

    # ------------------------------
    # 6. PCA
    # ------------------------------
    pca = None
    if apply_pca:
        print(f"\n6. Applying PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"   PCA reduced dimensions: {X_train_pca.shape[1]} components")
        print(f"   Total explained variance: {explained_var:.2f}%")

        X_train = pd.DataFrame(X_train_pca, index=X_train.index,
                               columns=[f"PC{i+1}" for i in range(X_train_pca.shape[1])])
        X_test = pd.DataFrame(X_test_pca, index=X_test.index,
                              columns=[f"PC{i+1}" for i in range(X_test_pca.shape[1])])

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70 + "\n")

    return X_train, X_test, y_train, y_test, pca, original_features

