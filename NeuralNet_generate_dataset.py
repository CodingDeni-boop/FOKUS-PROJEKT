import torch
import numpy as np
import pandas as pd
import os
from pipeline_code.NeuralNet_generate_features import triangulate
from pipeline_code.NeuralNet_generate_features import features
from pipeline_code.NeuralNet_generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits

X_path = "./pipeline_saved_processes/dataframes/NeuralNetDataframes_Lite/X"
y_path = "./pipeline_saved_processes/dataframes/NeuralNetDataframes_Lite/y"

### checks if X and y already exists, and if not, they get computed

features_collection = triangulate(
    collection_path = "./pipeline_inputs/collection",
    fps = 30,

    rescale_points = ("tr","tl"),
    rescale_distance = 0.64,
    filter_threshold = 0.9,
    construction_points = {"mid" : {"between_points" : ("tl", "tr", "bl", "br"), "mouse_or_oft" : "oft"},},
    smoothing = True,
    smoothing_mouse = 3,
    smoothing_oft = 20
    )

X : pd.DataFrame = features(features_collection, output_path = X_path,

            distance = {("neck","earl") : ("x","y","z"),
                        ("neck","earr") : ("x","y","z"),
                        ("neck","bcl") : ("x","y","z")},
            f_b_fill = True,

            embedding_length = list(range(-15,16))
                )

y = labels(labels_path = "./pipeline_inputs/labels",
        )

X = drop_non_analyzed_videos(X = X,y = y)
X, y = drop_last_frame(X = X, y = y)
X, y = drop_nas(X = X,y = y)
X = reduce_bits(X)


X = pd.read_csv("./pipeline_saved_processes/dataframes/X_lite.csv", index_col= ("video_id", "frame"))
for video_name in X.index.levels[0]:
    print(video_name)