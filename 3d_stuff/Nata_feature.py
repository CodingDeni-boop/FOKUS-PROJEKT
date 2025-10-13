import pandas as pd
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from sklearn import calibration

options = opt(fps=30)

with open('oft_tracking/Empty_Cage/collection_test/video_1_3dset/calibration.json') as f:
   calibration = json.load(f)

test = mv.from_yolo3r({"left_output": "oft_tracking/Empty_Cage/collection_test/video_1_3dset/left_output.csv",
                        "right_output":"oft_tracking/Empty_Cage/collection_test/video_1_3dset/right_output.csv"},"1_Empty_Cage_multiview",
                      options, calibration)

test3d = test.stereo_triangulate()

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/",options,py3r.TrackingMV)

# only 1 video, how to do for several?????????

test3dcol_tri = test3dcol.stereo_triangulate()

################################## MY STUFF ############################################################################

test3dcol_tri.strip_column_names()

# test3dcol_tri is TrackingCollection
features_collection = py3r.FeaturesCollection.from_tracking_collection(test3dcol_tri)

# Calculate azimuth from 'nose' to 'neck' across all tracking objects
azimuth_results = features_collection.azimuth('nose', 'neck')
print(f"Calculated azimuth for {len(azimuth_results)} tracking objects")

# Store all azimuth results
features_collection.store(azimuth_results)

# Check results
for handle, features_obj in features_collection.features_dict.items():
    print(f"\n{handle} now has features:")
    print(f"  - Azimuth data shape: {features_obj.data['azimuth_from_nose_to_neck'].shape}")


###### Access dataframe for model training
available_keys = list(features_collection.features_dict.keys())
print(f"Available keys: {available_keys}")

features_obj = features_collection.features_dict['video_1_3dset']

# Feature data as a DataFrame
feature_data = features_obj.data['azimuth_from_nose_to_neck']

# Classification Model
from sklearn.ensemble import RandomForestClassifier
from Random_tools.deepethogram_vid_import import labels_vid1

rf = RandomForestClassifier()
X = feature_data  # features
y = labels_vid1   # target labels
rf.fit(X, y)