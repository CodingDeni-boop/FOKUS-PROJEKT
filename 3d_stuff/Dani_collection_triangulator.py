import os
import shutil
import pandas as pd
import numpy as np
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
from py3r.behaviour.tracking.tracking_collection import TrackingCollection as Collection
from py3r.behaviour.tracking.tracking_mv import TrackingMV as MultiView
from py3r.behaviour.features.features_collection import FeaturesCollection as FeaturesCollection

collection_folder_path_src = "oft_tracking/Empty_Cage/collection_after_preprocessing/"
options = opt(fps=30)

"""
for file in os.listdir(collection_folder_path_src): 
    left = pd.read_csv(collection_folder_path_src+file+"/left.csv")
    right = pd.read_csv(collection_folder_path_src+file+"/right.csv")
    print(file,left.shape, right.shape)
"""
tracking_collection = Collection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_after_preprocessing",options,MultiView)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("hipl","hipr", 0.05, dims = ("x","y","z"))

"""
triangulated_tracking_collection.plot(trajectories=["bodycentre"], dims=("x", "y", "z"))
"""

pairs_of_points = pd.DataFrame({
    "point1": ["nose", "nose", "neck", "neck", "neck", "neck", "bcl", "bcr", "hipl", "hipr"],
    "point2": ["earl", "earr", "earl", "earr", "bcl", "bcr", "hipl", "hipr", "tailbase", "tailbase"]
})

collection = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)
for i in range(0,pairs_of_points.shape[0]):
    collection.distance_between(pairs_of_points.iloc[i,0],pairs_of_points.iloc[i,1],dims=("x","y","z")).store()
print(collection[0].data)

