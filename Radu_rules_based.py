import py3r
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV
#from py3r.behaviour.util.docdata import data_path
#lucreaza intai cu videorile 6 si 13
import pandas as pd
import glob
import os
from torch.nn.functional import threshold


#from natsort import natsorted

#from HGB_Pipe_2D import tracking_collection

###################################################################################################################################

def triangulate(collection_path: str,
                fps: int,
                rescale_points: tuple[str, str],
                rescale_distance: float,
                filter_threshold: float = 0.9,
                construction_points: dict[str: dict["between_points": tuple[str], "mouse_or_oft": str]] = None,
                smoothing=True,
                smoothing_mouse=3,
                smoothing_oft=20):
    options = opt(fps=fps)
    tracking_collection = TrackingCollection.from_yolo3r_folder(collection_path, options, TrackingMV)

    # Likelihood filter

    tracking_collection.filter_likelihood(filter_threshold)

    # Triangulation

    triangulated_tracking_collection = tracking_collection.stereo_triangulate()
    triangulated_tracking_collection.strip_column_names()
    triangulated_tracking_collection.rescale_by_known_distance(rescale_points[0], rescale_points[1], rescale_distance, dims=("x", "y", "z"))

    # Initialize smoothing

    smoothing_overrides1 = [
        # mouse
        (['nose'], 'mean', smoothing_mouse),
        (['headcentre'], 'mean', smoothing_mouse),
        (['neck'], 'mean', smoothing_mouse),
        (['earl'], 'mean', smoothing_mouse),
        (['earr'], 'mean', smoothing_mouse),
        (['bodycentre'], 'mean', smoothing_mouse),
        (['bcl'], 'mean', smoothing_mouse),
        (['bcr'], 'mean', smoothing_mouse),
        (['hipl'], 'mean', smoothing_mouse),
        (['hipr'], 'mean', smoothing_mouse),
        (['tailbase'], 'mean', smoothing_mouse),
        (['tailcentre'], 'mean', smoothing_mouse),
        (['tailtip'], 'mean', smoothing_mouse),
        # oft
        (['tl'], 'median', smoothing_oft),
        (['tr'], 'median', smoothing_oft),
        (['bl'], 'median', smoothing_oft),
        (['br'], 'median', smoothing_oft),
        (['top_tl'], 'median', smoothing_oft),
        (['top_tr'], 'median', smoothing_oft),
        (['top_bl'], 'median', smoothing_oft),
        (['top_br'], 'median', smoothing_oft)
    ]

    if smoothing:
        triangulated_tracking_collection.each.smooth_all(overrides = smoothing_overrides1)

    features_collection = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

    return features_collection

###################################################################################################################################

def features(features_collection: FeaturesCollection,
             azimuth: set[tuple[str, str]] = [],
             distance: set[tuple[str, str]] = [],
             distance_to_boundary : tuple[str] = [],
             speed: tuple = [],
             distance_change: tuple[str] = [],
             f_b_fill=True,
             #embedding_length=list(range(0, 1))
             ):
    # Azimuth/Azimuth_dev
    for handle in azimuth:
        features_collection.each.azimuth(handle[0], handle[1]).store()


    # Distances between points
    for handle in distance:
        features_collection.each.distance_between(handle[0], handle[1], dims=("x", "y", "z")).store()

    # Distance(s) to OFT boundary
    b=features_collection.each.define_static_boundary(['tl', 'tr', 'bl', 'br'], name='the_floor')
    for point in distance_to_boundary:
        features_collection.each.distance_to_boundary(point, boundary=b).store()

    # Speeds of points
    for point in speed:
        features_collection.each.speed(point, dims=("x", "y", "z")).store()

    # Absolute movement(s) of points
    for point in distance_change:
        features_collection.each.distance_change(point, dims=("x", "y", "z")).store()

    #print(dir(features_collection))

    #print(type(features_collection))
    ############################################### Missing data handling

    if f_b_fill:
        print("Missing data filling (forward/backward)...")

        # Forward fill then backward fill missing data
        for file in features_collection.keys():
            feature_obj = features_collection[file]
            df = feature_obj.data

            # Forward fill, then backward fill remaining NAs
            df = df.ffill().bfill()

            feature_obj.data = df

    #print("Embedding...")

    #embedding = {}
    #for column in features_collection[0].data.columns:
        #embedding[column] = list(embedding_length)
    #features_collection = features_collection.embedding_df(embedding)

    keys = list(features_collection.features_dict.keys())
    values = [v.data for v in features_collection.features_dict.values()]

    combined_features = pd.concat(values, keys=keys, names=['video_id', 'frame'])
    return combined_features
    # Extract features
    #feature_dict = {}
    #for handle in natsorted(features_collection):
        #feature_obj = features_collection[handle]
        #feature_dict[handle] = feature_obj

############################################################ MAIN #################################################################

tracking_collection = TrackingCollection.from_yolo3r_folder(folder_path = "./pipeline_inputs/collection", fps = 30, tracking_cls = TrackingMV)
#tracking_collection.filter_likelihood(threshold=0.9)
tri_tracking_collection = tracking_collection.stereo_triangulate()

#print("Tracking collection keys:", list(tracking_collection.keys()))
#print("Triangulated keys:", list(tri_tracking_collection.keys()))

tri_tracking_collection.each.strip_column_names()
tri_tracking_collection.each.rescale_by_known_distance("tr", "tl", 0.64, dims=("x", "y", "z"))

smoothing = True

smoothing_overrides = [
    #mouse
    (['nose'], 'mean', 3),
    (['headcentre'], 'mean', 3),
    (['neck'], 'mean', 3),
    (['earl'], 'mean', 3),
    (['earr'], 'mean', 3),
    (['bodycentre'], 'mean', 3),
    (['bcl'], 'mean', 3),
    (['bcr'], 'mean', 3),
    (['hipl'], 'mean', 3),
    (['hipr'], 'mean', 3),
    (['tailbase'], 'mean', 3),
    (['tailcentre'], 'mean', 3),
    (['tailtip'], 'mean', 3),
    #oft
    (['tl'], 'median', 3),
    (['tr'], 'median', 3),
    (['bl'], 'median', 3),
    (['br'], 'median', 3),
    (['top_tl'], 'median', 3),
    (['top_tr'], 'median', 3),
    (['top_bl'], 'median', 3),
    (['top_br'], 'median', 3)
]

if smoothing:
    tri_tracking_collection.each.smooth_all(overrides = smoothing_overrides)

features_collection = FeaturesCollection.from_tracking_collection(tri_tracking_collection)

#print("Collection keys at creation:", list(features_collection.keys()))
#print("Collection length at creation:", len(features_collection))
#print("Internal dict:", features_collection._obj_dict)

#features_collection.distance_between("nose", "bodycentre", dims=("x", "y", "z")).store(name="dist_nose_bodycentre")
#features_collection.azimuth_deviation("neck", "nose", "bodycentre", dims=("x", "y", "z")).store()

main_features = features(features_collection,
        azimuth={
            ("tailbase", "bodycentre"),
            ("bodycentre", "neck"),
            ("hipr", "bcr"),
            ("hipl", "bcl")
        },

        distance={
            ("hipl", "hipr"),
            ("earl", "earr"),
            ("nose", "earl"),
            ("nose", "earr")
        },

        distance_to_boundary=("nose",),

        speed=(
            "nose",
            "earl",
            "earr"
        ),

        distance_change=("bodycentre",),

        f_b_fill=False, #For now, because i want to be able to use nose NAs
)

#main_features.to_csv('raduman/shakira.csv', index=False)

#print("Keys:", list(main_features.keys()))
#print("Values:", list(main_features.values))

#print(main_features["speed_of_earl_in_xyz"][('13', 137)])

labels_6 = pd.read_csv("./pipeline_inputs/labels/6.csv")
labels_13 = pd.read_csv("./pipeline_inputs/labels/13.csv")

labels_6["video_id"] = "6"
labels_13["video_id"] = "13"

labels_6["frame"] = labels_6.index
labels_13["frame"] = labels_13.index

labels = pd.concat([labels_6, labels_13], axis=0)
labels = labels.set_index(["video_id", "frame"])

print (main_features.columns)

#WE DO DIAGNOSTICS AND EXPERIMENTATION NOW YEAH MFER WE CRAZY

all_features=main_features.columns

supprear_values = pd.DataFrame(columns = all_features)
unsupprear_values = pd.DataFrame(columns = all_features)
grooming_values = pd.DataFrame(columns = all_features)

vid = '6'
#for video 13
#framecount = 18063
#for video 6
framecount = 18095

for frame in range(framecount):
    for feature in all_features:
        supprear_values.loc[frame, feature] = main_features[feature][(vid, frame)] * labels["supportedrear"][(vid, frame)]
        unsupprear_values.loc[frame, feature] = main_features[feature][(vid, frame)] * labels["unsupportedrear"][(vid, frame)]
        grooming_values.loc[frame, feature] = main_features[feature][(vid, frame)] * labels["grooming"][(vid, frame)]

supprear_values.to_csv('raduman/supp_6.csv', index=False)
unsupprear_values.to_csv('raduman/unsupp_6.csv', index=False)
grooming_values.to_csv('raduman/groom_6.csv', index=False)























'''
        #embedding_length=list(range(-15, 16, 3))
    )

    y = labels(labels_path="./pipeline_inputs/labels", )

    X, y = drop_non_analyzed_videos(X=X, y=y)
    X, y = drop_last_frame(X=X, y=y)
    X = reduce_bits(X)
    X.to_csv(X_path)
    y.to_csv(y_path)

else:
    X = pd.read_csv(X_path, index_col=["video_id", "frame"])
    y = pd.read_csv(y_path, index_col=["video_id", "frame"])

if os.path.isfile(X_filtered_path):
    X = pd.read_csv(X_filtered_path, index_col=["video_id", "frame"])

else:
    X = collinearity_filter(X, threshold=0.95)
    X.to_csv(X_filtered_path)
'''




