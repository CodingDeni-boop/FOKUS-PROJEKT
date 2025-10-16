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



def get_vector(data, point1: str, point2: str, dims=("x", "y", "z")):
    v = pd.DataFrame()
    for dim in dims:
        v[dim]=data[point1 + "." + dim] - data[point2 + "." + dim]
    return v

def seg_angle(data, point1: str, point2: str, point3: str, point4: str):
    v = get_vector(data, point1, point2)
    lv = np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
    u = get_vector(data, point3, point4)
    lu = np.sqrt(u['x']**2 + u['y']**2 + u['z']**2)
    dot = v['x']*u['x'] + v['y']*u['y'] + v['z']*u['z']
    angle = pd.Series(np.arccos(dot/(lv*lu)))
    return angle