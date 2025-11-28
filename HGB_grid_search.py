from pipeline_code.generate_features import triangulate
from pipeline_code.generate_features import features
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.model_tools import video_train_test_split
from pipeline_code.filter_and_preprocess import scale
from pipeline_code.filter_and_preprocess import collinearity_then_uvfs
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_code.Shelf import Shelf
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from pipeline_code.model_tools import predict_multiIndex
from sklearn.model_selection import GridSearchCV
import joblib as job
import time
import pandas as pd
from natsort import natsorted
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from pipeline_code.PerformanceEvaluation import evaluate_model
import json
import numpy as np


# THE AIM OF THIS IS TO MAKE THIS A SINGLE FILE, WHICH USES OUR REPOSITORY AS A SORT OF LIBRARY.
# THIS IS AN ATTEMPT TO MAKE ORDER

start = time.time()

X_path = "./pipeline_saved_processes/dataframes/X_lite.csv"
y_path = "./pipeline_saved_processes/dataframes/y.csv"
model_path = "./pipeline_saved_processes/models/lite_HGB_grid.pkl"

### checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

    features_collection = triangulate(
        collection_path="./pipeline_inputs/collection",
        fps=30,

        rescale_points=("tr", "tl"),
        rescale_distance=0.64,
        filter_threshold=0.9,
        construction_points={"mid": {"between_points": ("tl", "tr", "bl", "br"), "mouse_or_oft": "oft"}, },
        smoothing=True,
        smoothing_mouse=3,
        smoothing_oft=20
    )

    X: pd.DataFrame = features(features_collection,

                               distance={("neck", "earl"): ("x", "y", "z"),
                                         ("neck", "earr"): ("x", "y", "z"),
                                         ("neck", "bcl"): ("x", "y", "z"),
                                         ("neck", "bcr"): ("x", "y", "z"),
                                         ("bcl", "hipl"): ("x", "y", "z"),
                                         ("bcr", "hipr"): ("x", "y", "z"),
                                         ("hipl", "tailbase"): ("x", "y", "z"),
                                         ("hipr", "tailbase"): ("x", "y", "z"),
                                         ("headcentre", "neck"): ("x", "y", "z"),
                                         ("neck", "bodycentre"): ("x", "y", "z"),
                                         ("bodycentre", "tailbase"): ("x", "y", "z"),
                                         ("headcentre", "earl"): ("x", "y", "z"),
                                         ("headcentre", "earr"): ("x", "y", "z"),
                                         ("bodycentre", "bcl"): ("x", "y", "z"),
                                         ("bodycentre", "bcr"): ("x", "y", "z"),
                                         ("bodycentre", "hipl"): ("x", "y", "z"),
                                         ("bodycentre", "hipr"): ("x", "y", "z"),
                                         ("headcentre", "mid"): ("z"),
                                         ("earl", "mid"): ("z"),
                                         ("earr", "mid"): ("z"),
                                         ("neck", "mid"): ("z"),
                                         ("bcl", "mid"): ("z"),
                                         ("bcr", "mid"): ("z"),
                                         ("bodycentre", "mid"): ("z"),
                                         ("hipl", "mid"): ("z"),
                                         ("hipr", "mid"): ("z"),
                                         ("tailcentre", "mid"): ("z")
                                         },

                               angle={("bodycentre", "neck", "neck", "headcentre"): "radians",
                                      ("bodycentre", "neck", "neck", "earl"): "radians",
                                      ("bodycentre", "neck", "neck", "earr"): "radians",
                                      ("tailbase", "bodycentre", "bodycentre", "neck"): "radians",
                                      ("tailbase", "bodycentre", "tailbase", "hipl"): "radians",
                                      ("tailbase", "bodycentre", "tailbase", "hipr"): "radians",
                                      ("tailbase", "bodycentre", "hipl", "bcl"): "radians",
                                      ("tailbase", "bodycentre", "hipr", "bcr"): "radians",
                                      ("bodycentre", "tailbase", "tailbase", "tailcentre"): "radians",
                                      ("bodycentre", "tailbase", "tailcentre", "tailtip"): "radians"
                                      },

                               speed=("headcentre",
                                      "earl",
                                      "earr",
                                      "neck",
                                      "bcl",
                                      "bcr",
                                      "bodycentre",
                                      "hipl",
                                      "hipr",
                                      "tailcentre"
                                      ),

                               distance_to_boundary=("headcentre",
                                                     "earl",
                                                     "earr",
                                                     "neck",
                                                     "bcl",
                                                     "bcr",
                                                     "bodycentre",
                                                     "hipl",
                                                     "hipr",
                                                     "tailcentre"
                                                     ),

                               is_point_recognized=(["nose"]),

                               volume={
                                   ("neck", "bodycentre", "bcl", "bcr"): ((0, 1, 2), (2, 1, 3), (0, 3, 1), (0, 2, 3)),
                                   ("bodycentre", "hipl", "tailbase", "hipr"): ((0, 3, 2), (3, 1, 2), (0, 2, 1),
                                                                                (0, 1, 3)),
                                   ("neck", "bcl", "hipl", "bodycentre"): ((0, 1, 3), (1, 2, 3), (3, 2, 0), (0, 2, 1)),
                                   ("neck", "bcr", "hipr", "bodycentre"): ((0, 3, 1), (1, 3, 2), (3, 0, 2), (0, 1, 2))
                                   },

                               standard_deviation=("headcentre.z",
                                                   "earl.z",
                                                   "earr.z",
                                                   "bodycentre.z",
                                                   "Volume_of_neck_bodycentre_bcl_bcr",
                                                   "Volume_of_bodycentre_hipl_tailbase_hipr",
                                                   "Volume_of_neck_bcl_hipl_bodycentre",
                                                   "Volume_of_neck_bcr_hipr_bodycentre"
                                                   ),

                               f_b_fill=True,

                               embedding_length=list(range(0, 1)),
                               )

    y = labels(labels_path="./pipeline_inputs/labels",
               )

    X = drop_non_analyzed_videos(X=X, y=y)
    X, y = drop_last_frame(X=X, y=y)
    X, y = drop_nas(X=X, y=y)
    X = reduce_bits(X)

    print("saving...")
    X.to_csv(X_path)
    y.to_csv(y_path)
    print("!files saved!")

else:

    X = pd.read_csv(X_path, index_col=["video_id", "frame"])
    y = pd.read_csv(y_path, index_col=["video_id", "frame"])

if not os.path.isfile(model_path):

    behaviours = ["background", "supportedrear", "unsupportedrear", "grooming"]

    # Load data

    X_train, X_test, y_train, y_test = video_train_test_split(X, y, 4, 42)
    X_train, X_test = scale(X_train, X_test)

    # Ravel
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Calculate class weights for multi-class imbalanced data
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Class distribution in training: {class_counts}")

    # For multi-class, calculate sample weights
    total_samples = len(y_train)
    n_classes = len(unique)
    class_weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train])
    print(f"Class weights: {class_weights}")

    # Undersample majority class
    #rus = RandomUnderSampler(sampling_strategy="majority")
    #X_train, y_train = rus.fit_resample(X_train, y_train)

    ### Tuned Model

    print("Histogram-Based Gradient Boosting Classifier")
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=100,  # equivalent to n_estimators
        max_depth=6,
        learning_rate=0.1,
        max_bins=255,  # default, can increase for more precision
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=False,
        verbose=0
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("With smoothing")
    evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=20)

    print("Without smoothing")
    evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=0)

    Shelf(X_train, X_test, model , model_path)

else:
    X_train, X_test, y_train, y_test, grid = Shelf.load(X, y, model_path)

X_path = "./pipeline_saved_processes/dataframes/X_no_emb.csv"
y_path = "./pipeline_saved_processes/dataframes/y.csv"
model_path = "./pipeline_saved_processes/models/no_emb_HGB_grid.pkl"


