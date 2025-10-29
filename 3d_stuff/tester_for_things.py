from math import isnan

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

# ride with the mob alhamdulillah

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
        for j in (i+1, len):
            df[j][col] = df[i][col]

        #now we go through and linearly fill gaps

        for pos in range(len):
            if (isnan(df[pos][col])): # if it is unknown
                startvalue = df[pos-1][col] # mark the last known value
                i = pos
                while (isnan(df[i][col])):
                    i += 1
                stopvalue = df[i][col] # mark the next known value
                step = (stopvalue - startvalue) / (i - pos + 1) # fit a line between startvalue and stopvalue
                i = pos
                while (isnan(df[i][col])):
                    df[i][col] = (i-pos+1)*step # fill
                    i += 1
                pos = i # go to where the next known value was. this makes it run in O(n)
    feature_obj.data = df


