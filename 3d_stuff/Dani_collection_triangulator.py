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

"""
triangulated_tracking_collection.plot(trajectories=["bodycentre"], dims=("x", "y", "z"))
"""

lines = pd.DataFrame({
    "point1": ["mouse_top.mouse_top_0.nose", "mouse_top.mouse_top_0.nose", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.bcl", "mouse_top.mouse_top_0.bcr", "mouse_top.mouse_top_0.hipl", "mouse_top.mouse_top_0.hipr"],
    "point2": ["mouse_top.mouse_top_0.earl", "mouse_top.mouse_top_0.earr", "mouse_top.mouse_top_0.earl", "mouse_top.mouse_top_0.earr", "mouse_top.mouse_top_0.bcl", "mouse_top.mouse_top_0.bcr", "mouse_top.mouse_top_0.hipl", "mouse_top.mouse_top_0.hipr", "mouse_top.mouse_top_0.tailbase", "mouse_top.mouse_top_0.tailbase"]
})

print(triangulated_tracking_collection[0].data)

collection = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

print(collection)

