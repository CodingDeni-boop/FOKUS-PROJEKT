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

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/", options, py3r.TrackingMV)

test3dcol_tri = test3dcol.stereo_triangulate()

test3dcol_tri.strip_column_names()

############################################## my code ##########################################################

# loop through each key to get tracking object
for recording in test3dcol_tri:
    tr = test3dcol_tri[recording]                   # TrackingMV object with .data and .meta
    print(f"\nProcessing {recording} ...")

    # get point names (like 'nose', 'tl', 'top_br', etc.)
    cols = list(tr.data.columns)
    points = sorted({".".join(c.split(".")[:-1]) for c in cols if not c.endswith(".likelihood")})

    # compute speed for each point
    features = feat(tr)
    speed_data = {}

    for p in points:
        dims = tuple(d for d in ("x", "y", "z") if f"{p}.{d}" in cols)
        res = features.speed(p, dims=dims)  # FeaturesResult
        speed_data[res.name] = res

speed_df = pd.DataFrame(speed_data)
print(speed_df)
print(speed_df.columns)

#take specific point, f.e. nose
nose_speed = speed_df['speed_of_nose_in_xyz']
