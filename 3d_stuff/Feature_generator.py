import pandas as pd
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv
from py3r.behaviour.features.features import Features as feat
from py3r.behaviour.features.features_collection import FeaturesCollection as featscoll
from py3r.behaviour.features.features_result import FeaturesResult as featsresult
from sklearn import calibration

options = opt(fps=30)

with open('oft_tracking/Empty_Cage/collection_test/video_1_3dset/calibration.json') as f:
   calibration = json.load(f)

test = mv.from_yolo3r({"left_output": "oft_tracking/Empty_Cage/collection_test/video_1_3dset/left_output.csv",
                        "right_output":"oft_tracking/Empty_Cage/collection_test/video_1_3dset/right_output.csv"},"1_Empty_Cage_multiview",
                      options, calibration)

test3d = test.stereo_triangulate()

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/",options,py3r.TrackingMV)

test3dcol_tri = test3dcol.stereo_triangulate()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##py3r.Tracking.rescale_by_known_distance

##test3dcol_tri.rescale_by_known_distance("tl","br", 0.64, dims = ("x","y","z"))
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## first make a dataframe that is a table with the bones on the lines and the point pairs on the columns
## when calculating jack skellington you will use that dataframe as a reference
## you will call a 3d distance_between on every pair, using numbers that you pull from test3dcol_tri
## the results of this go into jack skellington, which has the frames on the lines and the bones on the columns

temp = {
    "point1": ["mouse_top.mouse_top_0.nose", "mouse_top.mouse_top_0.nose", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.neck", "mouse_top.mouse_top_0.bcl", "mouse_top.mouse_top_0.bcr", "mouse_top.mouse_top_0.hipl", "mouse_top.mouse_top_0.hipr"],
    "point2": ["mouse_top.mouse_top_0.earl", "mouse_top.mouse_top_0.earr", "mouse_top.mouse_top_0.earl", "mouse_top.mouse_top_0.earr", "mouse_top.mouse_top_0.bcl", "mouse_top.mouse_top_0.bcr", "mouse_top.mouse_top_0.hipl", "mouse_top.mouse_top_0.hipr", "mouse_top.mouse_top_0.tailbase", "mouse_top.mouse_top_0.tailbase"]
}

bone_ends = pd.DataFrame(temp)

coll = {}
coll["skele"] = featscoll.from_tracking_collection(test3dcol_tri)

##and now we work with coll["skele"].data



