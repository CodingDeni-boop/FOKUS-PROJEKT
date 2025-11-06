

########################################## ACCURACY & CLASSIFICATION REPORT ############################################
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def smooth_predictions(predictions, min_frames):
    """
    Remove behavior prediction outliers shorter than min_frames.

        predictions: Array of predicted labels
        min_frames: Minimum number of consecutive frames for a behavior to be valid

    Returns:
        Smoothed predictions array
    """
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

def evaluate_model(model : object, X_train : pd.DataFrame(), y_train: pd.DataFrame(), X_test: pd.DataFrame(), y_test: pd.DataFrame(), min_frames=None ):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Apply smoothing to remove outliers below 20 frames
    y_pred = smooth_predictions(y_pred, min_frames)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    #print(f"Cross-validation score: {cross_val_score(model, X_train, y_train, cv=5).mean():.4f}")

    print("\n=== Classification Report - TRAIN SET ===")
    y_train_pred = model.predict(X_train)
    y_train_pred = smooth_predictions(y_train_pred, min_frames )
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(classification_report(y_train, y_train_pred))

    print("\n=== Classification Report - TEST SET ===")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    ################################### CONFUSION MATRIX ###############################################################

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in cells
        fmt='d',              # Format as integers
        cmap='Blues',         # Color scheme
        xticklabels=model.classes_,  # Label x-axis with class names
        yticklabels=model.classes_,  # Label y-axis with class names
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('Eval_output/confusion_matrix.png', dpi=300, bbox_inches='tight')

    ########################################### Feature Importance #########################################################
"""
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n=== Feature Importance ===")
        print(importance_df)
    else:
        print("\n=== Feature Importance ===")
        print("Feature names not available")
"""