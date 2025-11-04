import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_tools import video_train_test_split, drop_non_analyzed_videos, drop_last_frame


def preprocess_data(
        features_file="processed_features.csv",
        labels_file="processed_labels.csv",
        missing_threshold=0.05,
        test_videos=1,
        random_state = 42
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
    # 1. Load processed data
    # ------------------------------
    print("\n1. Loading processed data...")
    X = pd.read_csv(features_file, index_col=["video_id", "frame"])
    y = pd.read_csv(labels_file, index_col=["video_id", "frame"])
    print(f"   Initial X shape: {X.shape}")

    # ------------------------------
    # 2. Train/Test split
    # ------------------------------
    print(f"\n3. Splitting data (test_videos={test_videos})...")
    X_train, X_test, y_train, y_test = video_train_test_split(X, y, test_videos=test_videos, random_state=random_state)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ------------------------------
    # 3. Scaling
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

