import py3r.behaviour as py3r
import numpy as np
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from tools_3d import get_vector
from tools_3d import seg_angle

options = opt(fps=30)

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/",options,py3r.TrackingMV)

test3dcol_tri = test3dcol.stereo_triangulate()

################################## TESTING GROUNDS ############################################################################

test3dcol_tri.strip_column_names()

# test3dcol_tri is TrackingCollection
fc = py3r.FeaturesCollection.from_tracking_collection(test3dcol_tri)

# attempting a volume calculation



