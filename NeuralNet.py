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
from pipeline_code.PerformanceEvaluation import evaluate_model
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler
from torch import nn
from sklearn.metrics import classification_report

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device = mps_device)
else:
    print ("MPS device not found.")

X_path = "./pipeline_saved_processes/dataframes/X_31.csv"
y_path = "./pipeline_saved_processes/dataframes/y_31.csv"
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


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(114*31, 1024*5)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(1024*5, 512*5)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(512*5, 512)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(512, 4)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = self.relu3(x)
        
        x = self.linear4(x)
        
        return x
    

network1 = NeuralNet().to(mps_device)
print(network1)

def train_loop(dataloader : DataLoader, network : NeuralNet, loss_fn : nn.CrossEntropyLoss, optimizer : torch.optim.RMSprop):
    size = len(dataloader.dataset)
 
    network.train()
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(mps_device), y.to(mps_device)

        pred = network(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_samples += X.shape[0]  # count actual samples processed

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{total_samples:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(mps_device), y.to(mps_device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return pred

epochs = 5
batch_size = 128
learning_rate = 1e-4
class_weights = torch.tensor([1.0, 8.0, 15.0, 15.0]).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.RMSprop(network1.parameters(), lr = learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, network1, loss_function, optimizer)
    test_loop(test_dataloader, network1, loss_function).cpu().numpy()
print("Done!")



network1.eval()
with torch.no_grad():
    X_test = torch.from_numpy(X_test.to_numpy(dtype = "float32"))
    X_test = X_test.to(mps_device)
    logits = network1(X_test)
    predictions = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test

    y_pred = np.empty_like(predictions, dtype = object)
    print(y_pred.size)
    y_pred[predictions == 0] = "background"
    y_pred[predictions == 1] = "supportedrear"
    y_pred[predictions == 2] = "unsupportedrear"
    y_pred[predictions == 3] = "grooming"

    print(classification_report(y_true, y_pred))
