
from sklearn.ensemble import RandomForestClassifier
#from labels.deepethogram_vid_import import all_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np




''' old

features_obj = fc.features_dict['video_1_3dset']

labels_vid1 = labels_vid1.iloc[1:] # delete frame 0 row

y_cat = pd.Categorical(labels_vid1)
print(f"Categories: {y_cat.categories}")
'''


rf = RandomForestClassifier()
y = pd.read_csv("../labels/labels.csv", index_col=0)
X = pd.read_csv("../model/features.csv", index_col=0)
X = X.iloc[:y.shape[0]]


#y = all_labels.values.ravel()   # target labels

### take seperate vidoes as test set
# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced' )

rf.fit(X_train, y_train)

# Predict

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Cross-validation score: {cross_val_score(rf, X_train, y_train, cv=5).mean():.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
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