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
    i=0
    for dim in dims:
        v[i]=data[point1 + "." + dim] - data[point2 + "." + dim]
        i=i+1
    return v

def seg_angle(data, point1: str, point2: str, point3: str, point4: str):
    v = get_vector(data, point1, point2)
    lv = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    u = get_vector(data, point3, point4)
    lu = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    dot = np.dot(v, u)
    angle = np.arccos(dot/(lv*lu))