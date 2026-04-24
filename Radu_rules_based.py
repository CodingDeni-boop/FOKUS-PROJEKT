import py3r
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV
#from py3r.behaviour.util.docdata import data_path
#lucreaza intai cu videorile 6 si 13
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import glob
import os

from sympy.physics.units import acceleration
from torch.nn.functional import threshold


#from natsort import natsorted

#from HGB_Pipe_2D import tracking_collection

###################################################################################################################################

def triangulate(collection_path: str,
                fps: int,
                rescale_points: tuple[str, str],
                rescale_distance: float,
                filter_threshold: float = 0.9,
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
             heights: tuple = [],
             distance: set[tuple[str, str]] = [],
             distance_to_boundary : tuple[str] = [],
             speed: tuple = [],
             acceleration: tuple = [],
             f_b_fill=True,
             #embedding_length=list(range(0, 1))
             ):

    '''
    # Azimuth/Azimuth_dev
    for handle in azimuth:
        features_collection.each.azimuth(handle[0], handle[1]).store()
    '''

    # Heights, calculated as vertical distance to a point we know is always visible and on the floor (eg. tailcentre)
    for point in heights:
        features_collection.each.distance_between("tailcentre", point, dims = ("y", "z")).store()

    # Distances between points
    for handle in distance:
        features_collection.each.distance_between(handle[0], handle[1], dims = ("x", "y", "z")).store()

    # Distance(s) to OFT boundary
    b = features_collection.each.define_static_boundary(['tl', 'tr', 'bl', 'br'], name = 'edge')
    for point in distance_to_boundary:
        features_collection.each.distance_to_boundary(point, boundary = b).store()

    # Speeds of points
    for point in speed:
        features_collection.each.speed(point, dims = ("x", "y", "z")).store()

    # Acceleration of points
    for point in acceleration:
        features_collection.each.acceleration(point, dims = ("x", "y", "z")).store()

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

#BOILERPLATE CODE
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

#DIAGNOSTIC/DEBUGGING PRINTS
#print("Collection keys at creation:", list(features_collection.keys()))
#print("Collection length at creation:", len(features_collection))
#print("Internal dict:", features_collection._obj_dict)

main_features = features(features_collection,
        heights = (
            "nose",
            "headcentre",
            "bodycentre",
        ),

        distance = {
            ("hipl", "hipr"),
            ("nose", "earl"),
            ("nose", "earr"),
            ("headcentre", "neck")
        },

        distance_to_boundary = ("nose",),

        speed = (
            "nose",
            "earl",
            "earr",
            "neck",
            "bodycentre"
        ),

        acceleration = (
            "nose",
            "earl",
            "earr",
            "neck",
            "bodycentre"
        ),

        f_b_fill=True,
)


#CODE FOR DOING INTERPRETATIVE DATA ANALYSIS
'''
labels_6 = pd.read_csv("./pipeline_inputs/labels/6.csv")
labels_13 = pd.read_csv("./pipeline_inputs/labels/13.csv")

labels_6["video_id"] = "6"
labels_13["video_id"] = "13"

labels_6["frame"] = labels_6.index
labels_13["frame"] = labels_13.index

labels = pd.concat([labels_6, labels_13], axis=0)
labels = labels.set_index(["video_id", "frame"])

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

main_features.loc['6'].to_csv('raduman/features_6.csv', index=False)
main_features.loc['13'].to_csv('raduman/features_13.csv', index=False)

labels_6 = pd.read_csv('pipeline_inputs/labels/6.csv')
feats_6 = main_features.loc['6']

labels_13 = pd.read_csv('pipeline_inputs/labels/13.csv')
feats_13 = main_features.loc['13']

suppr_6 = labels_6["supportedrear"]
unsup_6 = labels_6["unsupportedrear"]
groom_6 = labels_6["grooming"]

suppr_13 = labels_13["supportedrear"]
unsup_13 = labels_13["unsupportedrear"]
groom_13 = labels_13["grooming"]

corr_suppr_6 = feats_6.apply(lambda col: col.corr(suppr_6))
corr_unsup_6 = feats_6.apply(lambda col: col.corr(unsup_6))
corr_groom_6 = feats_6.apply(lambda col: col.corr(groom_6))

corr_suppr_13 = feats_13.apply(lambda col: col.corr(suppr_13))
corr_unsup_13 = feats_13.apply(lambda col: col.corr(unsup_13))
corr_groom_13 = feats_13.apply(lambda col: col.corr(groom_13))

print("Correlates for video 6########################################################################")
print("SUPPORTED~REARING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(corr_suppr_6.sort_values(ascending=False))
print()
print("UNSUPPORTED~REARING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(corr_unsup_6.sort_values(ascending=False))
print()
print("GROOMING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(corr_groom_6.sort_values(ascending=False))
print()
print()

print("Correlates for video 13########################################################################")
print("SUPPORTED~REARING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(corr_suppr_13.sort_values(ascending=False))
print()
print("UNSUPPORTED~REARING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(corr_unsup_13.sort_values(ascending=False))
print()
print("GROOMING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(corr_groom_13.sort_values(ascending=False))
print()

#CODE FOR ACTUALLY CLASSIFYING BEHAVIOR
#collection["feature"][('video', frame)]
'''
videos = ['6', '13']

for video in videos:
    framecount = main_features["speed_of_nose_in_xyz"][video].shape[0]
    framecount = int(framecount)
    print(framecount)
    frame=0

    labels = pd.DataFrame(columns=["", "background", "supportedrear", "unsupportedrear", "grooming"])
    labels = labels.astype(int)  # labels starts filled with 0s

    for frame in range(framecount):
        label = "background"  # could become "supportedrear" or "unsupportedrear" or "grooming"

        #CHECK FOR A REAR
        if (main_features["distance_between_tailcentre_and_bodycentre_in_yz"][(video, frame)] >= 0.065): # body high enough?
            if (main_features["distance_between_tailcentre_and_nose_in_yz"][(video, frame)] >= 0.1): # head high enough?
                if (main_features["distance_to_boundary_static_nose_in_edge"][(video, frame)] < 0.1): # are we sniffing the wall?
                    label = "supportedrear"
                #elif (frame != 0 and labels["supportedrear"][frame-1] == 1):
                    #label = "supportedrear"
                else: # alright unsupported it is
                    label = "unsupportedrear"
            elif (frame != 0 and labels["supportedrear"][frame-1] == 1 and main_features["speed_of_nose_in_xyz"][(video, frame)] >= 0.07):
                label = "supportedrear" # if head low but moving fast, and body still high, we might be in the end frames of the rear
            elif (frame != 0 and labels["unsupportedrear"][frame-1] == 1 and main_features["speed_of_nose_in_xyz"][(video, frame)] >= 0.07):
                label = "unsupportedrear" # same edgecase check but for unsupported rear

        #CHECK FOR GROOMING, SKIP IF REAR FOUND
        #if (label == "background"):
            #GROOMCHECK
        labels.loc[frame, ""] = frame
        labels.loc[frame, "background"] = int(label == "background")
        labels.loc[frame, "supportedrear"] = int(label == "supportedrear")
        labels.loc[frame, "unsupportedrear"] = int(label == "unsupportedrear")
        labels.loc[frame, "grooming"] = int(label == "grooming")
    frame = frame + 1
    labels.loc[frame, ""] = frame
    labels.loc[frame, "background"] = 0
    labels.loc[frame, "supportedrear"] = 0
    labels.loc[frame, "unsupportedrear"] = 0
    labels.loc[frame, "grooming"] = 0
    file_path = "raduman/" + video + ".csv"
    labels.to_csv(file_path, index=False)
'''

#CODE FOR MAKING A COOL CONFUSION MATRIX
'''
pred = pd.read_csv('raduman/6.csv')
true = pd.read_csv('pipeline_inputs/labels/6.csv')

cols = ["background", "supportedrear", "unsupportedrear", "grooming"]

y_true = true[cols].idxmax(axis=1)
y_pred = pred[cols].idxmax(axis=1)

cm = confusion_matrix(y_true, y_pred, labels=cols, normalize="true")

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=cols, yticklabels=cols)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Man")
plt.tight_layout()
plt.show()
'''