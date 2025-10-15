import pandas as pd
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV

options = opt(fps=30)

tracking_collection = TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options, TrackingMV)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("hipl","hipr", 0.05, dims = ("x","y","z"))

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

# Azimuth
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

#Area of Mouse

# Extract Features

# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file].data
    feature_dict[file] = feature_obj

combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])
print(combined_features)
