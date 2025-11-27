from pipeline_code.generate_features import triangulate
from pipeline_code.generate_features import features
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.filter_and_preprocess import Filter
from pipeline_code.model_wrapper import ModelWrapper
from pipeline_code.model_wrapper import GridWrapper
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import joblib as job
import time
import pandas as pd
from natsort import natsorted
import os

# THE AIM OF THIS IS TO MAKE THIS A SINGLE FILE, WHICH USES OUR REPOSITORY AS A SORT OF LIBRARY. 
# THIS IS AN ATTEMPT TO MAKE ORDER

start = time.time()

X_path = "./pipeline_saved_processes/dataframes/X_lite.csv"
y_path = "./pipeline_saved_processes/dataframes/y.csv"
SVM_grid_search_path = "./pipeline_saved_processes/models/SVM_grid_search.pkl"


### checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

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

    X : pd.DataFrame = features(features_collection, 

                distance = {("neck","earl") : ("x","y","z"),
                            ("neck","earr") : ("x","y","z"),
                            ("neck","bcl") : ("x","y","z"),
                            ("neck","bcr") : ("x","y","z"),
                            ("bcl","hipl") : ("x","y","z"),
                            ("bcr","hipr") : ("x","y","z"),
                            ("hipl","tailbase") : ("x","y","z"),
                            ("hipr","tailbase") : ("x","y","z"),
                            ("headcentre","neck") : ("x","y","z"),
                            ("neck","bodycentre") : ("x","y","z"),
                            ("bodycentre","tailbase") : ("x","y","z"),
                            ("headcentre","earl") : ("x","y","z"),
                            ("headcentre","earr") : ("x","y","z"),
                            ("bodycentre","bcl") : ("x","y","z"),
                            ("bodycentre","bcr") : ("x","y","z"),
                            ("bodycentre","hipl") : ("x","y","z"),
                            ("bodycentre","hipr") : ("x","y","z"),
                            ("headcentre","mid") : ("x","y","z"),
                            ("earl","mid") : ("z"),
                            ("earr","mid") : ("z"),
                            ("neck","mid") : ("z"),
                            ("bcl","mid") : ("z"),
                            ("bcr","mid") : ("z"),
                            ("bodycentre","mid")  : ("z"),
                            ("hipl","mid") : ("z"),
                            ("hipr","mid") : ("z"),
                            ("tailcentre","mid") : ("z")
                        },
                        
                angle = {("bodycentre","neck","neck","headcentre") : "radians",
                            ("bodycentre","neck","neck","earl") : "radians",
                            ("bodycentre","neck","neck","earr") : "radians",
                            ("tailbase","bodycentre","bodycentre","neck") : "radians",
                            ("tailbase","bodycentre","tailbase","hipl") : "radians",
                            ("tailbase","bodycentre","tailbase","hipr") : "radians",
                            ("tailbase","bodycentre","hipl","bcl") : "radians",
                            ("tailbase","bodycentre","hipr","bcr") : "radians",
                            ("bodycentre","tailbase","tailbase","tailcentre") : "radians",
                            ("bodycentre","tailbase","tailcentre","tailtip") : "radians"
                        },
                        
                speed = ("headcentre", 
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

                distance_to_boundary = ("headcentre", 
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

                is_point_recognized = (["nose"]),
                
                volume = {("neck", "bodycentre", "bcl", "bcr") : ((0, 1, 2), (2, 1, 3), (0, 3, 1), (0, 2, 3)),
                        ("bodycentre", "hipl", "tailbase", "hipr") : ((0, 3, 2), (3, 1, 2), (0, 2 , 1), (0, 1, 3)),
                        ("neck", "bcl", "hipl", "bodycentre") : ((0, 1, 3), (1, 2, 3), (3, 2, 0), (0, 2, 1)),
                        ("neck", "bcr", "hipr", "bodycentre") : ((0, 3, 1), (1, 3, 2), (3, 0, 2), (0, 1, 2))
                        },
                
                standard_deviation = ("headcentre.z",
                                    "earl.z",
                                    "earr.z",
                                    "bodycentre.z",
                                    "Volume_of_neck_bodycentre_bcl_bcr",
                                    "Volume_of_bodycentre_hipl_tailbase_hipr",
                                    "Volume_of_neck_bcl_hipl_bodycentre",
                                    "Volume_of_neck_bcr_hipr_bodycentre"
                                    ),
                
                f_b_fill = True,

                embedding_length = list(range(-15,16)),
                    )

    y = labels(labels_path = "./pipeline_inputs/labels",
            )
    
    X = drop_non_analyzed_videos(X = X,y = y)
    X, y = drop_last_frame(X = X, y = y)
    X, y = drop_nas(X = X,y = y)

    print("saving...")
    X.to_csv(X_path)
    y.to_csv(y_path)
    print("!files saved!")

else:

    X = pd.read_csv(X_path, index_col=["video_id", "frame"])
    y = pd.read_csv(y_path, index_col=["video_id", "frame"])

X = reduce_bits(X)
filter = Filter(VarianceThreshold(threshold=0.2))
filter.fit(X)
filter.get_variance()
filter.graph_variance("./pipeline_outputs/VarianceThresholdGraph.png")
X = filter.transform(X)

if not os.path.isfile(SVM_grid_search_path):

    pipe = Pipeline([
            ("SVM", SVC(class_weight="balanced", probability=True))
        ])

    param_grid=[
        {
            "SVM__kernel": ["linear"],
            "SVM__C": [0.01,0.1,1, 10],
        },
        {
            "SVM__kernel": ["poly"],
            "SVM__C": [0.01,0.1,1, 10],
            "SVM__degree": [2,3,4, 5, 9],
            "SVM__coef0": [0,1],
        },
        {
            "SVM__kernel": ["rbf"],
            "SVM__C": [0.01,0.1,1, 10],
            "SVM__gamma" : [1, 0.1, 0.01, 0.001, 0.0001]
        },
        {
            "SVM__kernel": ["sigmoid"],
            "SVM__C": [0.01,0.1,1, 10],
            "SVM__coef0": [0, 1],
        }
                    ]

    wrapped_SVM_grid = GridWrapper(X = X, y = y,
                    estimator = pipe,
                    param_grid = param_grid,
                    scoring = "f1_macro",
                    cv = 5,
                    n_jobs = 4,
                    train_test_test_videos = 3, 
                    random_state = 42, 
                    scaling = True, 
                    undersampling = True,
                    labels = ("background", "supportedrear", "unsupportedrear", "grooming"))

    wrapped_SVM_grid.fit_grid()
    wrapped_SVM_grid.predict()
    wrapped_SVM_grid.save(SVM_grid_search_path)
else:
    wrapped_SVM_grid = GridWrapper.load(SVM_grid_search_path, X, y)

wrapped_SVM_grid.evaluate()


    #job.dump(features, f"pipeline_saved_processes/Feature_Collection_{int(time.localtime()[2])}-{int(time.localtime()[1])}-{int(time.localtime()[0])}_{int(time.localtime()[3])}:{int(time.localtime()[4])}")


end = time.time()
print("time elapsed:", f"{int((end-start)//3600)}h {int(((end-start)%3600)//60)}m {int((end-start)%60)}s")

