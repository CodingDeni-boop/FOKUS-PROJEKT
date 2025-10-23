import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model_tools import video_train_test_split, drop_non_analyzed_videos, drop_last_frame


def preprocess_data(
        features_file="features.csv",
        labels_file="nataliia_labels.csv",
        missing_threshold=0.1,
        test_videos=2,
        apply_pca=True,
        n_components=0.95  # keep 95% variance by default
):
    """
    Load data, handle missing values, split into train/test sets, scale, and optionally apply PCA.

    Parameters
    ----------
    features_file : str
        Path to features CSV file.
    labels_file : str
        Path to labels CSV file.
    missing_threshold : float
        Maximum percentage of missing values allowed per column (0.1 = 10%).
    test_videos : int
        Number of videos to use for test set.
    apply_pca : bool
        Whether to apply PCA dimensionality reduction after scaling.
    n_components : int or float
        Number of components for PCA or explained variance ratio (e.g., 0.95 keeps 95% variance).

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays or DataFrames
        Data ready for model training.
    pca : PCA object or None
        Trained PCA object (if apply_pca=True), else None.
    """

    print("=" * 70)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    y = pd.read_csv(labels_file, index_col=["video_id", "frame"])
    X = pd.read_csv(features_file, index_col=["video_id", "frame"])
    X = drop_non_analyzed_videos(X, y)
    X, y = drop_last_frame(X, y)
    print(f"   Initial X shape: {X.shape}")

    # Handle missing data
    print(f"\n2. Handling missing data (column threshold: {missing_threshold * 100:.0f}%)...")
    na_percentage = X.isna().mean()
    columns_to_keep = na_percentage[na_percentage <= missing_threshold].index
    columns_dropped = na_percentage[na_percentage > missing_threshold].index
    print(f"   Dropped {len(columns_dropped)} columns (> {missing_threshold * 100:.0f}% missing)")
    X = X[columns_to_keep]

    # Drop rows with any remaining NaN
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"   Final X shape after cleaning: {X.shape}")

    # Train/Test split
    print(f"\n3. Splitting data (test_videos = {test_videos})...")
    X_train, X_test, y_train, y_test = video_train_test_split(X, y, test_videos=test_videos)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print(f"   Train set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")

    # Scaling
    print("\n4. Scaling numeric features...")
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    print("   Scaling complete.")

    # PCA
    pca = None
    if apply_pca:
        print(f"\n5. Applying PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train[num_features])
        X_test_pca = pca.transform(X_test[num_features])

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"   PCA reduced dimensions: {X_train_pca.shape[1]} components")
        print(f"   Total explained variance: {explained_var:.2f}%")

        # Replace numeric columns with PCA components
        X_train_pca_df = pd.DataFrame(X_train_pca,
                                      index=X_train.index,
                                      columns=[f"PC{i + 1}" for i in range(X_train_pca.shape[1])])
        X_test_pca_df = pd.DataFrame(X_test_pca,
                                     index=X_test.index,
                                     columns=[f"PC{i + 1}" for i in range(X_test_pca.shape[1])])

        X_train = X_train_pca_df
        X_test = X_test_pca_df

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70 + "\n")

    return X_train, X_test, y_train, y_test, pca, X.columns.tolist()

