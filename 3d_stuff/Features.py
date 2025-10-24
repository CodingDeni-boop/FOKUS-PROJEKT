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

# Add construction point middle of OFT

triangulated_tracking_collection.construction_point("mid",["tl","tr","bl","br"],dims=("x","y","z"))

# Interpolating and Smoothing

triangulated_tracking_collection.interpolate(limit = 3)

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
    "tr": {"window": 20, "type": "median"},
    "tl": {"window": 20, "type": "median"},
    "br": {"window": 20, "type": "median"},
    "bl": {"window": 20, "type": "median"},
    "top_tr": {"window": 20, "type": "median"},
    "top_tl": {"window": 20, "type": "median"},
    "top_br": {"window": 20, "type": "median"},
    "top_bl": {"window": 20, "type": "median"},
    "mid": {"window": 20, "type": "median"}
})

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

# Distance

pairs_of_points_for_lines = pd.DataFrame({
    "point1": ["nose", "nose", "neck", "neck", "neck", "neck", "bcl",  "bcr",  "hipl",     "hipr",     "nose",       "headcentre", "neck",       "bodycentre", "headcentre", "headcentre", "bodycentre", "bodycentre", "bodycentre", "bodycentre"],
    "point2": ["earl", "earr", "earl", "earr", "bcl",  "bcr",  "hipl", "hipr", "tailbase", "tailbase", "headcentre", "neck",       "bodycentre", "tailbase",   "earl",       "earr",       "bcl",        "bcr",        "hipl",       "hipr"]
})

for i in range(0,pairs_of_points_for_lines.shape[0]):
    fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "x").store()
    fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "y").store()
    fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "z").store()

print("distance calculated and stored")

# Azimuth / Angles

pairs_of_points_for_angles = pd.DataFrame({
    "point1": ["bodycentre","bodycentre","bodycentre","tailbase",   "tailbase",  "tailbase",  "tailbase",  "tailbase",  "bodycentre","bodycentre"],
    "point2": ["neck",      "neck",       "neck",      "bodycentre","bodycentre","bodycentre","bodycentre","bodycentre","tailbase","tailbase"],
    "point3": ["neck",      "neck",       "neck",      "bodycentre","tailbase","tailbase",    "hipl",       "hipr",     "tailbase","tailcentre"],
    "point4": ["headcentre","earl",       "earr",      "neck",      "hipl",     "hipr",       "bcl",        "bcr",      "tailcentre","tailtip"]
})

for i in range(0,pairs_of_points_for_angles.shape[0]):
    #fc.angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
    fc.sin_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
    fc.cos_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
    #fc.angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z"))
    fc.sin_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z")).store()
    fc.cos_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z")).store()

print("angle calculated and stored")

# Speed

first_F = next(iter(fc.features_dict.values()))
cols = first_F.tracking.data.columns

for col in cols:
    if col.endswith(".x"):
        p = col[:-2]
        fc.speed(p, dims=("x","y","z")).store()

print("Speed calculated and stored")


#Distances to boundary

all_relevant_points = ("nose", "headcentre", "earl", "earr", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailcentre")
for point in all_relevant_points:
    fc.distance_to_boundary_dynamic(point, ["tl", "tr", "bl", "br"], "oft").store()

print("Distance to boundary calculated and stored")

#Heights

for point in all_relevant_points:
    fc.height(point).store()

print("height calculated and stored")

# is it BALL?

fc.is_recognized("nose").store()
fc.is_recognized("tailbase").store()

#Standard deviation
fc.standard_dev("headcentre").store()

#Volume

fc.volume(points = ["neck", "bodycentre", "bcl", "bcr"], faces = [[0, 1, 2], [2, 1, 3], [0, 3, 1], [0, 2, 3]]).store()
fc.volume(points = ["bodycentre", "hipl", "tailbase", "hipr"], faces = [[0, 3, 2], [3, 1, 2], [0, 2 , 1], [0, 1, 3]]).store()
fc.volume(points = ["neck", "bcl", "hipl", "bodycentre"], faces = [[0, 1, 3], [1, 2, 3], [3, 2, 0], [0, 2, 1]]).store()
fc.volume(points = ["neck", "bcr", "hipr", "bodycentre"], faces = [[0, 3, 1], [1, 3, 2], [3, 0, 2], [0, 1, 2]]).store()

print("Volume calculated and stored")

#Embed
embedding = {}
for column in fc[0].data.columns:
    embedding[column] =  [-3,-2,-1,0,1,2,3]
fc = fc.embedding_df(embedding)




print("Embedding done")

# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file]
    feature_dict[file] = feature_obj

combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])


print("saving...")

combined_features.to_csv("./../model/features.csv")

print("!file saved!")



