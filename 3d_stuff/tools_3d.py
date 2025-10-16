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



def get_vector(self, point1: str, point2: str, dims=("x", "y", "z")):
    v = pd.DataFrame()
    v['x'] = self.tracking.data[point1+".x"]-self.tracking.data[point2+".x"]
    v['y'] = self.tracking.data[point1+".y"]-self.tracking.data[point2+".y"]
    v['z'] = self.tracking.data[point1+".z"]-self.tracking.data[point2+".z"]
    return v


def seg_angle(self, point1: str, point2: str, point3: str, point4: str) -> FeaturesResult:
    v = get_vector(self, point1, point2)
    u = get_vector(self, point3, point4)
    lv = np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
    lu = np.sqrt(u['x']**2 + u['y']**2 + u['z']**2)
    dotman = v['x']*u['x'] + v['y']*u['y'] + v['z']*u['z']
    angle = np.arccos(dotman/(lv*lu))
    name = f"angle_{point1} {point2}_to_ {point3} {point4}"


