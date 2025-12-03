

########################################## ACCURACY & CLASSIFICATION REPORT ############################################
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import numpy as np


def count_behavior_instances(predictions, behavior_label):
    """
    Count the number of behavior instances by detecting transitions to the behavior.
    A new instance is counted each time the behavior starts (transition from another label to this label).

    Args:
        predictions: Array of predicted labels
        behavior_label: The specific behavior to count

    Returns:
        Number of instances of the behavior
    """

    # Create binary array: 1 where prediction matches behavior, 0 otherwise
    behavior_mask = (predictions == behavior_label).astype(int)

    # Detect transitions: count where diff > 0.5 (i.e., transition from 0 to 1)
    changes = np.diff(behavior_mask, prepend=0)
    instances = (changes > 0.5).sum()

    return instances

def smooth_predictions(predictions, min_frames):
    
    #Remove behavior prediction outliers shorter than min_frames.

       # predictions: Array of predicted labels
      #  min_frames: Minimum number of consecutive frames for a behavior to be valid

    #Returns:
      #  Smoothed predictions array
    
    if len(predictions) < min_frames:
        return predictions

    smoothed = predictions.copy()
    i = 0

    while i < len(smoothed):
        current_label = smoothed[i]
        # Find the length of the current behavior segment
        segment_start = i
        while i < len(smoothed) and smoothed[i] == current_label:
            i += 1
        segment_length = i - segment_start

        # If segment is shorter than min_frames, replace with neighboring behavior
        if segment_length < min_frames:
            # Determine replacement label from neighbors
            prev_label = smoothed[segment_start - 1] if segment_start > 0 else None
            next_label = smoothed[i] if i < len(smoothed) else None

            # Choose the label that appears in neighbors (prefer previous)
            if prev_label is not None and next_label is not None and prev_label == next_label:
                replacement = prev_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label
            else:
                # Keep original if no neighbors available
                continue

            # Replace the short segment
            smoothed[segment_start:i] = replacement

    return smoothed

def evaluate_model(model : BaseEstimator, X_train : pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, min_frames=None, conf_matrix_path : str = "pipeline_outputs/conf_matrix.png" ):
    y_pred = model.predict(X_test)

    # Apply smoothing to remove outliers below 20 frames
    if min_frames is not None:
        y_pred = smooth_predictions(y_pred, min_frames)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    #print(f"Cross-validation score: {cross_val_score(model, X_train, y_train, cv=5).mean():.4f}")

    print("\n=== Classification Report - TRAIN SET ===")
    y_train_pred = model.predict(X_train)
    if min_frames is not None:
        y_train_pred = smooth_predictions(y_train_pred, min_frames )
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(classification_report(y_train, y_train_pred))

    print("\n=== Classification Report - TEST SET ===")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    ################################### CONFUSION MATRIX ###############################################################

    print("\n=== Normalised Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    print(cmn)

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cmn,
        annot=True,           # Show numbers in cells
        fmt='g',              # Format as integers
        cmap='Blues',         # Color scheme
        xticklabels=model.classes_,  # Label x-axis with class names
        yticklabels=model.classes_,  # Label y-axis with class names
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    ################################### BEHAVIOR INSTANCE COUNTS ###############################################################

    print("\n=== Behavior Instance Counts ===")
    behaviors = np.unique(y_test)

    print(f"{'Behavior':<20} {'True Count':<15} {'Predicted Count':<15}")
    print("="*50)
    for behavior in behaviors:
        true_count = count_behavior_instances(y_test, behavior)
        pred_count = count_behavior_instances(y_pred, behavior)
        print(f"{behavior:<20} {true_count:<15} {pred_count:<15}")


