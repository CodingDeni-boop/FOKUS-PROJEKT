

########################################## ACCURACY & CLASSIFICATION REPORT ############################################

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Cross-validation score: {cross_val_score(rf, X_train, y_train, cv=5).mean():.4f}")

print("\n=== Classification Report - TRAIN SET ===")
y_train_pred = rf.predict(X_train)
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(classification_report(y_train, y_train_pred))

print("\n=== Classification Report - TEST SET ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

################################### CONFUSION MATRIX ###################################################################

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
    xticklabels=rf.classes_,  # Label x-axis with class names
    yticklabels=rf.classes_,  # Label y-axis with class names
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('Eval_output/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

########################################### Feature Importance #########################################################

if hasattr(X, 'columns'):
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Feature Importance ===")
    print(importance_df)
else:
    print("\n=== Feature Importance ===")
    print("Feature names not available")
