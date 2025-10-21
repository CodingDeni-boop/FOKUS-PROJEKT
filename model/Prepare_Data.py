"""
Data loading and preprocessing pipeline
"""
import pandas as pd
from model_tools import video_train_test_split, drop_non_analyzed_videos, drop_last_frame
import numpy as np


def load_and_prepare_data(features_file="features.csv", 
                          labels_file="nataliia_labels.csv",
                          missing_threshold=0.1,
                          test_videos=2):
    """
    Load data, handle missing values, and split into train/test sets.
    
    Parameters:
    -----------
    features_file : str
        Path to features CSV file
    labels_file : str
        Path to labels CSV file
    missing_threshold : float
        Maximum percentage of missing values allowed per column (0.1 = 10%)
    test_videos : int
        Number of videos to use for test set
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames/arrays
        Split data ready for training
    """
    
    print("="*70)
    print("DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    y = pd.read_csv(labels_file, index_col=["video_id", "frame"])
    X = pd.read_csv(features_file, index_col=["video_id", "frame"])
    X = drop_non_analyzed_videos(X, y)
    X, y = drop_last_frame(X, y)
    print(f"   Initial X shape: {X.shape}")
    
    # Handle missing data
    print(f"\n2. Handling missing data (threshold: {missing_threshold*100}%)...")
    na_percentage = X.isna().mean()
    columns_to_keep = na_percentage[na_percentage <= missing_threshold].index
    columns_dropped = na_percentage[na_percentage > missing_threshold].index
    
    print(f"   Dropped {len(columns_dropped)} columns with >{missing_threshold*100}% missing values")
    if len(columns_dropped) > 0 and len(columns_dropped) <= 10:
        print(f"   Dropped columns: {columns_dropped.tolist()}")
    
    X = X[columns_to_keep]
    
    # Remove rows with any missing values
    valid_mask = X.notna().all(axis=1)
    valid_X = X[valid_mask]
    valid_y = y[valid_mask]
    
    print(f"   Removed {len(X) - len(valid_X)} rows with missing values")
    print(f"   Final X shape: {valid_X.shape}")
    
    # Train/Test split
    print(f"\n3. Splitting data (test_videos: {test_videos})...")
    X_train, X_test, y_train, y_test = video_train_test_split(
        valid_X, valid_y, test_videos=test_videos
    )
    
    # Convert labels to numpy arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    print(f"   Train set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")

    # Scaling
    from sklearn.preprocessing import StandardScaler

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    sc = StandardScaler()
    X_train[num_features] = sc.fit_transform(X_train[num_features])
    X_test[num_features] = sc.transform(X_test[num_features])
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70 + "\n")
    
    return X_train, X_test, y_train, y_test
