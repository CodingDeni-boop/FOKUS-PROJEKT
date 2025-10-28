

########################################## ACCURACY & CLASSIFICATION REPORT ############################################
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model : object, X_train : pd.DataFrame(), y_train: pd.DataFrame(), X_test: pd.DataFrame(), y_test: pd.DataFrame()):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Cross-validation score: {cross_val_score(model, X_train, y_train, cv=5).mean():.4f}")

    print("\n=== Classification Report - TRAIN SET ===")
    y_train_pred = model.predict(X_train)
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