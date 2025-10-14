
from sklearn.ensemble import RandomForestClassifier
from Random_tools.deepethogram_vid_import import labels_vid1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from Nata_features import *

# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file].data
    feature_dict[file] = feature_obj

features_obj = fc.features_dict['video_1_3dset']

"""
labels_vid1 = labels_vid1.iloc[1:] # delete frame 0 row


y_cat = pd.Categorical(labels_vid1)
print(f"Categories: {y_cat.categories}")
"""


rf = RandomForestClassifier()
X = feature_data  # features
y = labels_vid1.values.ravel()   # target labels


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
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(importance_df)