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
all_relevant_points = ("nose", "headcentre", "earl", "earr", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailbase")
tracking_collection = TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options, TrackingMV)

# Likelihood filter

filter_threshold = 0.9
tracking_collection.filter_likelihood(filter_threshold)

#Triangulation

triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("tr","tl", 0.64, dims = ("x","y","z"))

# Add construction point middle of OFT

triangulated_tracking_collection.construction_point("mid",["tl","tr","bl","br"],dims=("x","y","z"))

# (Interpolating) and Smoothing

# triangulated_tracking_collection.interpolate(limit = 3)

smoothing_mouse = 3
smoothing_oft = 20

triangulated_tracking_collection.smooth({

    # mouse
    "nose": {"window": smoothing_mouse, "type": "mean"},
    "headcentre": {"window": smoothing_mouse, "type": "mean"},
    "neck": {"window": smoothing_mouse, "type": "mean"},
    "earl": {"window": smoothing_mouse, "type": "mean"},
    "earr": {"window": smoothing_mouse, "type": "mean"},
    "bodycentre": {"window": smoothing_mouse, "type": "mean"},
    "bcl": {"window": smoothing_mouse, "type": "mean"},
    "bcr": {"window": smoothing_mouse, "type": "mean"},
    "hipl": {"window": smoothing_mouse, "type": "mean"},
    "hipr": {"window": smoothing_mouse, "type": "mean"},
    "tailbase": {"window": smoothing_mouse, "type": "mean"},
    "tailcentre": {"window": smoothing_mouse, "type": "mean"},
    "tailtip": {"window": smoothing_mouse, "type": "mean"},

    # oft
    "tr": {"window": smoothing_oft, "type": "median"},
    "tl": {"window": smoothing_oft, "type": "median"},
    "br": {"window": smoothing_oft, "type": "median"},
    "bl": {"window": smoothing_oft, "type": "median"},
    "top_tr": {"window": smoothing_oft, "type": "median"},
    "top_tl": {"window": smoothing_oft, "type": "median"},
    "top_br": {"window": smoothing_oft, "type": "median"},
    "top_bl": {"window": smoothing_oft, "type": "median"},
    "mid": {"window": smoothing_oft, "type": "median"}
})

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

# Distance

print("calculating distance...")

pairs_of_points_for_lines = pd.DataFrame({
    "point1": ["neck", "neck", "neck", "neck", "bcl",  "bcr",  "hipl",     "hipr",     "headcentre",  "neck",       "bodycentre", "headcentre", "headcentre", "bodycentre", "bodycentre", "bodycentre", "bodycentre"],
    "point2": ["earl", "earr", "bcl",  "bcr",  "hipl", "hipr", "tailbase", "tailbase",  "neck",       "bodycentre", "tailbase",   "earl",       "earr",       "bcl",        "bcr",        "hipl",       "hipr"]
})

for i in range(0,pairs_of_points_for_lines.shape[0]):
    fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "x").store()
    fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "y").store()
    fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "z").store()

# Azimuth / Angles

print("calculating angles...")

pairs_of_points_for_angles = pd.DataFrame({
    "point1": ["bodycentre","bodycentre","bodycentre","tailbase",   "tailbase",  "tailbase",  "tailbase",  "tailbase",  "bodycentre","bodycentre"],
    "point2": ["neck",      "neck",       "neck",      "bodycentre","bodycentre","bodycentre","bodycentre","bodycentre","tailbase","tailbase"],
    "point3": ["neck",      "neck",       "neck",      "bodycentre","tailbase","tailbase",    "hipl",       "hipr",     "tailbase","tailcentre"],
    "point4": ["headcentre","earl",       "earr",      "neck",      "hipl",     "hipr",       "bcl",        "bcr",      "tailcentre","tailtip"]
})

for i in range(0,pairs_of_points_for_angles.shape[0]):
    fc.angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
    #fc.sin_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
    #fc.cos_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
    fc.angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z"))
    #fc.sin_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z")).store()
    #fc.cos_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z")).store()

# Speed

print("calculating speed...")

"""
first_F = next(iter(fc.features_dict.values()))
cols = first_F.tracking.data.columns

for col in cols:
    if col.endswith(".x"):
        p = col[:-2]
        fc.speed(p, dims=("x","y","z")).store()
"""

for point in all_relevant_points:
     fc.speed(point, dims=("x","y","z")).store()

#Distances to boundary

print("calculating distance to boundary...")

all_relevant_points = ("headcentre", "earl", "earr", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailcentre")
for point in all_relevant_points:
    fc.distance_to_boundary_dynamic(point, ["tl", "tr", "bl", "br"], "oft").store()


#Heights

print("calculating height...")

for point in all_relevant_points:
    fc.height(point).store()

# is it BALL?

print("calculating ball...")

fc.is_recognized("nose").store()
fc.is_recognized("tailbase").store()

#Standard deviation
print("calculating standard deviation...")

fc.standard_dev("headcentre").store()

#Volume

print("calculating volume...")

fc.volume(points = ["neck", "bodycentre", "bcl", "bcr"], faces = [[0, 1, 2], [2, 1, 3], [0, 3, 1], [0, 2, 3]]).store()
fc.volume(points = ["bodycentre", "hipl", "tailbase", "hipr"], faces = [[0, 3, 2], [3, 1, 2], [0, 2 , 1], [0, 1, 3]]).store()
fc.volume(points = ["neck", "bcl", "hipl", "bodycentre"], faces = [[0, 1, 3], [1, 2, 3], [3, 2, 0], [0, 2, 1]]).store()
fc.volume(points = ["neck", "bcr", "hipr", "bodycentre"], faces = [[0, 3, 1], [1, 3, 2], [3, 0, 2], [0, 1, 2]]).store()



############################################### Missing data handling

print("Missing data filling (forward/backward)...")

# Forward fill then backward fill missing data
"""for file in fc.keys():
    feature_obj = fc[file]
    df = feature_obj.data

    # Forward fill, then backward fill remaining NAs
    df = df.ffill().bfill()

    feature_obj.data = df"""

# Linear fill of missing data
for file in fc.keys():
    feature_obj = fc[file]
    df = feature_obj.data

    len = df.shape[0]
    ncol = df.shape[1]

    for col in range(ncol):
        # first we make sure theres no nas at the start or at the end

        i = 0
        while (isnan(df[i][col])):
            i += 1
        for j in range(i):
            df[j][col] = df[i][col]
        i = len
        while (isnan(df[i][col])):
            i -= 1
        for j in (i + 1, len):
            df[j][col] = df[i][col]

        # now we go through and linearly fill gaps

        for pos in range(len):
            if (isnan(df[pos][col])):  # if it is unknown
                startvalue = df[pos - 1][col]  # mark the last known value
                i = pos
                while (isnan(df[i][col])):
                    i += 1
                stopvalue = df[i][col]  # mark the next known value
                step = (stopvalue - startvalue) / (i - pos + 1)  # fit a line between startvalue and stopvalue
                i = pos
                while (isnan(df[i][col])):
                    df[i][col] = (i - pos + 1) * step  # fill
                    i += 1
                pos = i  # go to where the next known value was. this makes it run in O(n)
    feature_obj.data = df



print("Embedding...")

#Embed
embedding = {}
for column in fc[0].data.columns:
    embedding[column] =  list(range(-15, 16))
fc = fc.embedding_df(embedding)

# Extract features
feature_dict = {}
for file in fc.keys():
    feature_obj = fc[file]
    feature_dict[file] = feature_obj

combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])


print("saving...")

combined_features.to_csv("./../model/features.csv")

print("!file saved!")



