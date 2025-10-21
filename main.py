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
import time




options = opt(fps=30)

tracking_collection = TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test",options, TrackingMV)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("tr","tl", 0.64, dims = ("x","y","z"))

triangulated_tracking_collection.interpolate()

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)



fc.volume(points = ["headcentre","nose","earl","earr"],faces = [[0,1,2],[0,2,3],[0,3,1],[1,3,2]])
