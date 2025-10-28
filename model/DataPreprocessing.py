import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_tools import video_train_test_split, drop_non_analyzed_videos, drop_last_frame


def preprocess_data(
        features_file="features.csv",
        labels_file="nataliia_labels.csv",
        missing_threshold=0.05,
        test_videos=2
):
    """
    Load data, handle missing values, split, and scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays or DataFrames
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

"""
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
"""

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

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70 + "\n")

    return X_train, X_test, y_train, y_test

