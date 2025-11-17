import pandas as pd
from math import isnan
import numpy as np
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV



def triangulate(collection_path : str, 
                fps : int, 
                rescale_points : tuple[str], 
                rescale_distance = float, 
                filter_threshold : int = 0.9,
                construction_points : dict[str : dict["between_points" : tuple[str], "mouse_or_oft" : str]] = None,
                smoothing = True,
                smoothing_mouse = 3,
                smoothing_oft = 20):
    

    options = opt(fps=fps)
    tracking_collection = TrackingCollection.from_yolo3r_folder(collection_path,options, TrackingMV)

    # Likelihood filter

    tracking_collection.filter_likelihood(filter_threshold)

    #Triangulation

    triangulated_tracking_collection = tracking_collection.stereo_triangulate()
    triangulated_tracking_collection.strip_column_names()
    triangulated_tracking_collection.rescale_by_known_distance(rescale_points[0],rescale_points[1], rescale_distance, dims = ("x","y","z"))

    # Initialize smoothing

    smoothing_dict = {
        # mouse
        "nose": {"window": smoothing_mouse, "type": "mean"},
        "headcentre": {"window": smoothing_mouse, "type": "mean"},
        "neck": {"window": smoothing_mouse, "type": "mean"},
        "earl": {"window": smoothing_mouse, "type": "mean"},
        "earr": {"window": smoothing_mouse, "type": "mean"},
        "bodycentre": {"window": smoothing_mouse, "type": "mean"},
        "bcl": {"window": smoothing_mouse, "type": "mean"},
        "bcr": {"window": smoothing_mouse, "type": "mean"},
        "hipl": {"window": smoothing_mouse, "type": "mean"},
        "hipr": {"window": smoothing_mouse, "type": "mean"},
        "tailbase": {"window": smoothing_mouse, "type": "mean"},
        "tailcentre": {"window": smoothing_mouse, "type": "mean"},
        "tailtip": {"window": smoothing_mouse, "type": "mean"},

        # oft
        "tr": {"window": smoothing_oft, "type": "median"},
        "tl": {"window": smoothing_oft, "type": "median"},
        "br": {"window": smoothing_oft, "type": "median"},
        "bl": {"window": smoothing_oft, "type": "median"},
        "top_tr": {"window": smoothing_oft, "type": "median"},
        "top_tl": {"window": smoothing_oft, "type": "median"},
        "top_br": {"window": smoothing_oft, "type": "median"},
        "top_bl": {"window": smoothing_oft, "type": "median"},
    }
    if not construction_points==None:
        for handle in construction_points:
            construction_infos = construction_points[handle]
            triangulated_tracking_collection.construction_point(handle,construction_infos["between_points"],dims=("x","y","z"))
            if construction_infos["mouse_or_oft"] == "mouse":
                smoothing_dict[handle] = {"window": smoothing_mouse, "type": "mean"}
            elif construction_infos["mouse_or_oft"] == "oft":
                smoothing_dict[handle] = {"window": smoothing_oft, "type": "median"}
            else:
                raise ValueError(f"{construction_infos['mouse_or_oft']} only accepts 'mouse' or 'oft' as values"  )
            print(f"Created construction point {handle} between {construction_infos['between_points']} as {construction_infos['mouse_or_oft']} point")
        
    if smoothing:
        triangulated_tracking_collection.smooth(smoothing_dict)

    fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

    return fc

    # Distance


def generate_features(features_collection : FeaturesCollection, 
                      distance = tuple[tuple],
                      ):
    
    all_relevant_points = ("nose", "headcentre", "earl", "earr", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailbase")
    

    print("calculating distance...")

    pairs_of_points_for_lines = pd.DataFrame({
        "point1": ["neck", "neck", "neck", "neck", "bcl",  "bcr",  "hipl",     "hipr",     "headcentre",  "neck",       "bodycentre", "headcentre", "headcentre", "bodycentre", "bodycentre", "bodycentre", "bodycentre"],
        "point2": ["earl", "earr", "bcl",  "bcr",  "hipl", "hipr", "tailbase", "tailbase",  "neck",       "bodycentre", "tailbase",   "earl",       "earr",       "bcl",        "bcr",        "hipl",       "hipr"]
    })

    for i in range(0,pairs_of_points_for_lines.shape[0]):
        fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "x").store()
        fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "y").store()
        fc.distance_on_axis(pairs_of_points_for_lines.iloc[i, 0], pairs_of_points_for_lines.iloc[i, 1], "z").store()

    # Azimuth / Angles

    print("calculating angles...")

    pairs_of_points_for_angles = pd.DataFrame({
        "point1": ["bodycentre","bodycentre","bodycentre","tailbase",   "tailbase",  "tailbase",  "tailbase",  "tailbase",  "bodycentre","bodycentre"],
        "point2": ["neck",      "neck",       "neck",      "bodycentre","bodycentre","bodycentre","bodycentre","bodycentre","tailbase","tailbase"],
        "point3": ["neck",      "neck",       "neck",      "bodycentre","tailbase","tailbase",    "hipl",       "hipr",     "tailbase","tailcentre"],
        "point4": ["headcentre","earl",       "earr",      "neck",      "hipl",     "hipr",       "bcl",        "bcr",      "tailcentre","tailtip"]
    })

    for i in range(0,pairs_of_points_for_angles.shape[0]):
        fc.angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
        #fc.sin_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
        #fc.cos_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("x","y")).store()
        fc.angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z"))
        #fc.sin_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z")).store()
        #fc.cos_of_angle(pairs_of_points_for_angles.iloc[i,0],pairs_of_points_for_angles.iloc[i,1],pairs_of_points_for_angles.iloc[i,2],pairs_of_points_for_angles.iloc[i,3],plane=("y","z")).store()

    # Speed

    print("calculating speed...")

    """
    first_F = next(iter(fc.features_dict.values()))
    cols = first_F.tracking.data.columns

    for col in cols:
        if col.endswith(".x"):
            p = col[:-2]
            fc.speed(p, dims=("x","y","z")).store()
    """

    for point in all_relevant_points:
        fc.speed(point, dims=("x","y","z")).store()

    #Distances to boundary

    print("calculating distance to boundary...")

    all_relevant_points = ("headcentre", "earl", "earr", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailcentre")
    for point in all_relevant_points:
        fc.distance_to_boundary_dynamic(point, ["tl", "tr", "bl", "br"], "oft").store()


    #Heights

    print("calculating height...")

    for point in all_relevant_points:
        fc.height(point).store()

    # is it BALL?

    print("calculating ball...")

    fc.is_recognized("nose").store()
    fc.is_recognized("tailbase").store()

    #Volume

    print("calculating volume...")

    fc.volume(points = ["neck", "bodycentre", "bcl", "bcr"], faces = [[0, 1, 2], [2, 1, 3], [0, 3, 1], [0, 2, 3]]).store()
    fc.volume(points = ["bodycentre", "hipl", "tailbase", "hipr"], faces = [[0, 3, 2], [3, 1, 2], [0, 2 , 1], [0, 1, 3]]).store()
    fc.volume(points = ["neck", "bcl", "hipl", "bodycentre"], faces = [[0, 1, 3], [1, 2, 3], [3, 2, 0], [0, 2, 1]]).store()
    fc.volume(points = ["neck", "bcr", "hipr", "bodycentre"], faces = [[0, 3, 1], [1, 3, 2], [3, 0, 2], [0, 1, 2]]).store()

    #Standard deviation
    print("calculating standard deviation...")

    fc.standard_dev("headcentre.z").store()
    fc.standard_dev("earl.z").store()
    fc.standard_dev("earr.z").store()
    fc.standard_dev("bodycentre.z").store()
    fc.standard_dev("Volume_of_neck_bodycentre_bcl_bcr").store()
    fc.standard_dev("Volume_of_bodycentre_hipl_tailbase_hipr").store()
    fc.standard_dev("Volume_of_neck_bcl_hipl_bodycentre").store()
    fc.standard_dev("Volume_of_neck_bcr_hipr_bodycentre").store()

    ############################################### Missing data handling

    print("Missing data filling (forward/backward)...")

    # Forward fill then backward fill missing data
    for file in fc.keys():
        feature_obj = fc[file]
        df = feature_obj.data

        # Forward fill, then backward fill remaining NAs
        df = df.ffill().bfill()

        feature_obj.data = df
    '''
    # Linear fill of missing data
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
            for j in (i + 1, len):
                df[j][col] = df[i][col]

            # now we go through and linearly fill gaps

            for pos in range(len):
                if (isnan(df[pos][col])):  # if it is unknown
                    startvalue = df[pos - 1][col]  # mark the last known value
                    i = pos
                    while (isnan(df[i][col])):
                        i += 1
                    stopvalue = df[i][col]  # mark the next known value
                    step = (stopvalue - startvalue) / (i - pos + 1)  # fit a line between startvalue and stopvalue
                    i = pos
                    while (isnan(df[i][col])):
                        df[i][col] = (i - pos + 1) * step  # fill
                        i += 1
                    pos = i  # go to where the next known value was. this makes it run in O(n)
        feature_obj.data = df
    '''


    print("Embedding...")

    #Embed
    embedding = {}
    for column in fc[0].data.columns:
        embedding[column] =  list(range(0, 1))
    fc = fc.embedding_df(embedding)

    # Extract features
    feature_dict = {}
    for file in fc.keys():
        feature_obj = fc[file]
        feature_dict[file] = feature_obj

    combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])


    print("saving...")

    combined_features.to_csv("./../model/features_lite.csv")

    print("!file saved!")



