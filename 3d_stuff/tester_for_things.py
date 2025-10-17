import py3r.behaviour as py3r
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

# only 1 video, how to do for several?????????

test3dcol_tri = test3dcol.stereo_triangulate()

################################## MY STUFF ############################################################################

test3dcol_tri.strip_column_names()

# test3dcol_tri is TrackingCollection
fc = py3r.FeaturesCollection.from_tracking_collection(test3dcol_tri)

# Calculate azimuth from 'nose' to 'neck' across all tracking objects
azimuth_results = fc.azimuth('neck', 'nose')
print(f"Calculated azimuth for {len(azimuth_results)} tracking objects")

# Store all azimuth results
fc.store(azimuth_results)

#trying to get angle between two segments, defined by 4 points
fc.angle('nose', 'neck', 'neck', 'bodycentre', ("x", "y")).store()
fc.angle('nose', 'neck', 'neck', 'bodycentre', ("y", "z")).store()


