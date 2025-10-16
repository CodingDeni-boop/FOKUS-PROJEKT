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
from 3d_tools import get_vector
from 3d_tools import seg_angle

options = opt(fps=30)

tracking_collection = TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options, TrackingMV)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("tr","tl", 0.64, dims = ("x","y","z"))

# Smoothing and interpolating

triangulated_tracking_collection.smooth({

    # mouse
    "nose": {"window": 5, "type": "mean"},
    "headcentre": {"window": 5, "type": "mean"},
    "neck": {"window": 5, "type": "mean"},
    "earl": {"window": 5, "type": "mean"},
    "earr": {"window": 5, "type": "mean"},
    "bodycentre": {"window": 5, "type": "mean"},
    "bcl": {"window": 5, "type": "mean"},
    "bcr": {"window": 5, "type": "mean"},
    "hipl": {"window": 5, "type": "mean"},
    "hipr": {"window": 5, "type": "mean"},
    "tailbase": {"window": 5, "type": "mean"},
    "tailcentre": {"window": 5, "type": "mean"},
    "tailtip": {"window": 5, "type": "mean"},

    # oft
    "tr": {"window": 5, "type": "mean"},
    "tl": {"window": 5, "type": "mean"},
    "br": {"window": 5, "type": "mean"},
    "bl": {"window": 5, "type": "mean"},
    "top_tr": {"window": 5, "type": "mean"},
    "top_tl": {"window": 5, "type": "mean"},
    "top_br": {"window": 5, "type": "mean"},
    "top_bl": {"window": 5, "type": "mean"}
})
triangulated_tracking_collection.interpolate()

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

# Azimuth / Angles
fc.azimuth('nose','neck').store()
fc.azimuth('neck', 'bodycentre').store()
fc.azimuth('headcentre', 'neck').store()

# Distance

pairs_of_points = pd.DataFrame({
    "point1": ["nose", "nose", "neck", "neck", "neck", "neck", "bcl", "bcr", "hipl", "hipr"],
    "point2": ["earl", "earr", "earl", "earr", "bcl", "bcr", "hipl", "hipr", "tailbase", "tailbase"]
})

for i in range(0,pairs_of_points.shape[0]):
    fc.distance_between(pairs_of_points.iloc[i,0],pairs_of_points.iloc[i,1],dims=("x","y","z")).store()
print(fc[0].data)

# Speed

first_F = next(iter(fc.features_dict.values()))
cols = first_F.tracking.data.columns

for col in cols:
    if col.endswith(".x"):
        p = col[:-2]
        fc.speed(p, dims=("x","y","z")).store()


#Area of Mouse

# Extract Features

# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file].data
    feature_dict[file] = feature_obj

combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])


combined_features.to_csv("./../model/features.csv")

