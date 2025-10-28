import pandas as pd
from model_tools import drop_non_analyzed_videos, drop_last_frame

def load_data(
        features_file="features.csv",
        labels_file="nataliia_labels.csv",
):

    print("=" * 70)
    print("DATA LOADING")
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
    # 2. Drop NA from embedding
    # ------------------------------

    valid_mask = X.isnotna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print("Finished dropping NA's")

    # ------------------------------
    # 3. Save processed data
    # ------------------------------
    print("\n2. Saving processed data...")
    X.to_csv("model/processed_features.csv")
    y.to_csv("model/processed_labels.csv")
    print("   Saved to processed_features.csv and processed_labels.csv")

    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETE")
    print("=" * 70 + "\n")

    return X, y