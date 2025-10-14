import pandas as pd
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult

options = opt(fps=30)

with open('oft_tracking/Empty_Cage/collection_test/video_1_3dset/calibration.json') as f:
   calibration = json.load(f)

test = mv.from_yolo3r({"left_output": "oft_tracking/Empty_Cage/collection_test/video_1_3dset/left_output.csv",
                        "right_output":"oft_tracking/Empty_Cage/collection_test/video_1_3dset/right_output.csv"},"1_Empty_Cage_multiview",
                      options, calibration)

test3d = test.stereo_triangulate()

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/",options,py3r.TrackingMV)

test3dcol_tri = test3dcol.stereo_triangulate()

test3dcol_tri.strip_column_names()

######################################### Features ##################################

fc = FeaturesCollection.from_tracking_collection(test3dcol_tri)


fc.azimuth('nose','neck').store()
fc.azimuth('neck', 'bodycentre').store()
fc.azimuth('headcentre', 'neck').store()


# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file].data
    feature_dict[file] = feature_obj
features_obj = fc.features_dict['video_1_3dset']

# Feature data as a DataFrame
"""
feature1 = features_obj.data['azimuth_from_nose_to_neck']
feature2 = features_obj.data['azimuth_from_neck_to_bodycentre']
feature3 = features_obj.data['azimuth_from_headcentre_to_neck']
"""

#feature_data = pd.concat([feature1, feature2, feature3], axis=1)
feature_data = features_obj.data

from sklearn.ensemble import RandomForestClassifier
from labels.deepethogram_vid_import import labels_vid1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

labels_vid1 = labels_vid1.iloc[1:] # delete frame 0 row

print("labels_vid1",labels_vid1)
"""
y_cat = pd.Categorical(labels_vid1)
print(f"Categories: {y_cat.categories}")
"""

print("missing data", feature_data.isnull().sum())

rf = RandomForestClassifier()
X = feature_data  # features
y = labels_vid1.values.ravel()   # target labels

print("This is x", X)

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