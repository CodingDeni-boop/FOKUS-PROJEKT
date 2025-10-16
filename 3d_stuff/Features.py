import pandas as pd
import numpy as np
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from tools_3d import get_vector
from tools_3d import seg_angle

options = opt(fps=30)

tracking_collection = TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options, TrackingMV)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("tr","tl", 0.64, dims = ("x","y","z"))

# Smoothing and interpolating

triangulated_tracking_collection.smooth({

    # mouse
    "nose": {"window": 3, "type": "mean"},
    "headcentre": {"window": 3, "type": "mean"},
    "neck": {"window": 3, "type": "mean"},
    "earl": {"window": 3, "type": "mean"},
    "earr": {"window": 3, "type": "mean"},
    "bodycentre": {"window": 3, "type": "mean"},
    "bcl": {"window": 3, "type": "mean"},
    "bcr": {"window": 3, "type": "mean"},
    "hipl": {"window": 3, "type": "mean"},
    "hipr": {"window": 3, "type": "mean"},
    "tailbase": {"window": 3, "type": "mean"},
    "tailcentre": {"window": 3, "type": "mean"},
    "tailtip": {"window": 3, "type": "mean"},

    # oft
    "tr": {"window": 35, "type": "mean"},
    "tl": {"window": 35, "type": "mean"},
    "br": {"window": 35, "type": "mean"},
    "bl": {"window": 35, "type": "mean"},
    "top_tr": {"window": 35, "type": "mean"},
    "top_tl": {"window": 35, "type": "mean"},
    "top_br": {"window": 35, "type": "mean"},
    "top_bl": {"window": 35, "type": "mean"}
})

triangulated_tracking_collection.interpolate()

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)


# Distance

pairs_of_points = pd.DataFrame({
    "point1": ["nose", "nose", "neck", "neck", "neck", "neck", "bcl",  "bcr",  "hipl",     "hipr",     "nose",       "headcentre", "neck",       "bodycentre", "headcentre", "headcentre", "bodycentre", "bodycentre", "bodycentre", "bodycentre"],
    "point2": ["earl", "earr", "earl", "earr", "bcl",  "bcr",  "hipl", "hipr", "tailbase", "tailbase", "headcentre", "neck",       "bodycentre", "tailbase",   "earl",       "earr",       "bcl",        "bcr",        "hipl",       "hipr"]
})

for i in range(0,pairs_of_points.shape[0]):
    fc.distance_between(pairs_of_points.iloc[i,0],pairs_of_points.iloc[i,1],dims=("x","y","z")).store()

# Azimuth / Angles
fc.azimuth('nose','neck').store()
fc.azimuth('neck', 'bodycentre').store()
fc.azimuth('headcentre', 'neck').store()

fc.angle('nose', 'neck', 'neck', 'bodycentre').store()

# Speed

first_F = next(iter(fc.features_dict.values()))
cols = first_F.tracking.data.columns

for col in cols:
    if col.endswith(".x"):
        p = col[:-2]
        fc.speed(p, dims=("x","y","z")).store()


#Area of Mouse

# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file].data
    feature_dict[file] = feature_obj

combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])
combined_features.to_csv("./../model/features.csv")

print(combined_features)
print("file saved:")