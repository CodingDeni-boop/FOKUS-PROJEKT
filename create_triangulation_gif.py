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

tracking_collection = TrackingCollection.from_yolo3r_folder("./pipeline_inputs/collection_for_gif",options, TrackingMV)
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


# Skeleton Tracking Animation
triangulated_tracking_collection.save_3d_tracking_video_multi_view(out_path = "./pipeline_outputs/tracking_gifs/video_9.gif", 
                                                                   startframe = 4161, endframe = 4250, 
                                                                   lines = [("tl","tr")
                                                                            ,("tr","br"),("br","bl"),("bl","tl"),("nose", "neck"),("tailbase","tailcentre"),("tailcentre","tailtip"),
                                                                                                                                                                          ("top_tr", "top_tl"), ("top_tl", "top_bl"), ("top_bl", "top_br"), ("top_br", "top_tr"), 
                                                                                                                                                                          ("neck", "bodycentre"), ("bodycentre", "tailbase")],
                                                                    
                                                                    dpi = 150, 
                                                                    line_color = (0, 0, 0, 0.6 ), 
                                                                    point_color = (0, 0, 1, 0.8))
