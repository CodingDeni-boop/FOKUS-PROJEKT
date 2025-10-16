
from sklearn.ensemble import RandomForestClassifier
#from labels.deepethogram_vid_import import all_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame

rf = RandomForestClassifier()
y = pd.read_csv("labels.csv", index_col=["video_id","frame"])
#y = y["label"]
X = pd.read_csv("features.csv", index_col=["video_id","frame"])
#X = X.drop(columns=["frame"])
X = drop_non_analyzed_videos(X,y)
y = drop_last_frame(X,y)


### take seperate vidoes as test set
# 1. Train/Test Split
X_train, X_test, y_train, y_test = video_train_test_split(
    X, y, test_videos=1)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced' )

rf.fit(X_train, y_train)

# Predict

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# Evaluates
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

