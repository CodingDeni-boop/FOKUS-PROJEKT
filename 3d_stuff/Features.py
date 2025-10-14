import pandas as pd
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking_collection import TrackingCollection as Collection
from py3r.behaviour.tracking.tracking_mv import TrackingMV as MultiView

collection_folder_path_src = "oft_tracking/Empty_Cage/collection/"
options = opt(fps=30)

tracking_collection = Collection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options,MultiView)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("hipl","hipr", 0.05, dims = ("x","y","z"))

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

# Azimuth
fc.azimuth('nose','neck').store()
fc.azimuth('neck', 'bodycentre').store()
fc.azimuth('headcentre', 'neck').store()

# Speed




# Distance

