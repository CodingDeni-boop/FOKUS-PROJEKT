import torch
import numpy as np
import pandas as pd
import os
from pipeline_code.generate_features import triangulate
from pipeline_code.generate_features import features
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.model_tools import video_train_test_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler
from torch import nn

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device = mps_device)
else:
    print ("MPS device not found.")

X_path = "./pipeline_saved_processes/dataframes/X_lite.csv"
y_path = "./pipeline_saved_processes/dataframes/y_lite.csv"
model_path = "./pipeline_saved_processes/models/SVM_lite_poly.pkl"
conf_matrix_path = "pipeline_outputs/SVM/SVM_lite_poly.png"

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
                            ("headcentre","mid") : ("z"),
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
    X = reduce_bits(X)

    print("saving...")
    X.to_csv(X_path)
    y.to_csv(y_path)
    print("!files saved!")

else:

    X = pd.read_csv(X_path, index_col=["video_id", "frame"])
    y = pd.read_csv(y_path, index_col=["video_id", "frame"])

X_train, X_test, y_train, y_test = video_train_test_split(X, y, 4, 42)
scaler = StandardScaler().set_output(transform = "pandas")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class OFTDataset(Dataset):

    def __init__(self, X : pd.DataFrame, y : pd.DataFrame, transform = None, target_transform = None):
        self.transform = transform
        self.target_transform = target_transform

            ##  TRANSFORM y SO THAT 0 = background, 1 = supportedrear, 2 = unsupportedrear, 3 = grooming ##
        y_coded = np.zeros_like(y)
        y_coded[y == "supportedrear"] = 1
        y_coded[y == "unsupportedrear"] = 2
        y_coded[y == "grooming"] = 3
        y_coded = y_coded.transpose().squeeze().astype("int8")
        del y

        self.X = torch.from_numpy(X.to_numpy(dtype = "float32"))
        self.y = torch.from_numpy(y_coded)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        frame_X = self.X[idx, :]
        frame_y = self.y[idx]

        if self.transform:
            frame_X = self.transform(frame_X)
        if self.target_transform:
            frame_y = self.target_transform(frame_y)

        return frame_X, frame_y


training_set = OFTDataset(X_train, y_train)
test_set = OFTDataset(X_test, y_test)
train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)        ## SHUFFLES FRAMES, TRY TO NOT SHUFFLE AND SEE IF CHANGES
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

train_features, train_label = next(iter(train_dataloader))
train_features : torch.Tensor
train_label : torch.Tensor

class NeuralNet(nn.Module)

