from pipeline_code.generate_features import triangulate
from pipeline_code.generate_features import features
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.model_wrapper import ModelWrapper
from pipeline_code.model_wrapper import GridWrapper
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib as job
import time
import pandas as pd
from natsort import natsorted
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
import os
import matplotlib.pyplot as plt

# THE AIM OF THIS IS TO MAKE THIS A SINGLE FILE, WHICH USES OUR REPOSITORY AS A SORT OF LIBRARY.
# THIS IS AN ATTEMPT TO MAKE ORDER

start = time.time()

X_path = "./pipeline_saved_processes/dataframes/X_lite.csv"
y_path = "./pipeline_saved_processes/dataframes/y.csv"
HGB_grid_search_path = "./pipeline_saved_processes/models/HGB_grid_search.pkl"

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

    X = features(features_collection,

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
                           ("headcentre", "mid"): ("x", "y", "z"),
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

                 is_point_recognized=("nose",
                                      "tailbase"
                                      ),

                 volume={("neck", "bodycentre", "bcl", "bcr"): ((0, 1, 2), (2, 1, 3), (0, 3, 1), (0, 2, 3)),
                         ("bodycentre", "hipl", "tailbase", "hipr"): ((0, 3, 2), (3, 1, 2), (0, 2, 1), (0, 1, 3)),
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

                 output_path=X_path
                 )

    y = labels(labels_path="./pipeline_inputs/labels",
               output_path=y_path
               )

else:

    X = pd.read_csv(X_path, index_col=["video_id", "frame"])

    y = pd.read_csv(y_path, index_col=["video_id", "frame"])

X = drop_non_analyzed_videos(X=X, y=y)
X, y = drop_last_frame(X=X, y=y)
X, y = drop_nas(X=X, y=y)

if not os.path.isfile(HGB_grid_search_path):

    param_grid = {
        'max_iter' : [100],  # equivalent to n_estimators
        'max_depth' : [6],
        'learning_rate' : [0.1],
        'max_bins' : [255],  # default, can increase for more precision
        'min_samples_leaf' : [20],
        'l2_regularization' : [0.0]
    }

    wrapped_HGB_grid = GridWrapper(X = X, y = y,
                    estimator = HGB(random_state=42, early_stopping=False, verbose=0),
                    param_grid = param_grid,
                    scoring = "f1_macro",
                    cv = 5,
                    n_jobs = 2,
                    train_test_test_videos =4,
                    random_state = 42,
                    scaling = True,
                    undersampling =False,
                    labels = ("background", "supportedrear", "unsupportedrear", "grooming"))

    sample_weights =wrapped_HGB_grid.class_weights()
    wrapped_HGB_grid.fit_grid(sample_weight=sample_weights)
    wrapped_HGB_grid.predict()
    wrapped_HGB_grid.save(HGB_grid_search_path)

else:
    wrapped_HGB_grid = GridWrapper.load(HGB_grid_search_path, X, y)

wrapped_HGB_grid.evaluate()

# Extract feature importances
importances = wrapped_HGB_grid.model.feature_importances_
feature_names = wrapped_HGB_grid.X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(100))


# Plot top 10 feature importances
top_n_plot = 10
top_features_plot = feature_importance_df.head(top_n_plot)
plt.figure(figsize=(10, 12))
plt.barh(range(top_n_plot), top_features_plot['Importance'], align='center')
plt.yticks(range(top_n_plot), top_features_plot['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
model_name =  "Histogram Gradient Boosting"
plt.title(f'Top {top_n_plot} {model_name} Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('pipeline_outputs/feature_importances_HGB.png', dpi=300, bbox_inches='tight')
plt.close()
end = time.time()