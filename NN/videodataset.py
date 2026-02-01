import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from utilities import terminal_colors
import numpy as np

class VideoDataset(Dataset):

    def __init__(self, path : str, video_names : list, scaler : StandardScaler, frames : int = 18000, 
                 behaviors : dict = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}, 
                 transform = None, target_transform = None, identity : str = "dataset"):
        
        self.transform = transform
        self.target_transform = target_transform
        self.path = path
        self.video_names = video_names
        self.scaler = scaler
        self.frames = frames
        self.behaviors = behaviors
        self.identity = identity

        print(terminal_colors.GREEN + f"{identity} initialized:\n" +
              terminal_colors.CYAN +"   videos = " + terminal_colors.ENDC + f"{self.video_names}\n"+
              terminal_colors.CYAN +"   scaler = " + terminal_colors.ENDC + f"{self.scaler}\n" +
              terminal_colors.CYAN +"   behaviors = " + terminal_colors.ENDC + f"{self.behaviors}\n"+
              terminal_colors.CYAN +"   frames = " + terminal_colors.ENDC + f"{self.frames}\n")

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        X = pd.read_csv(self.path + f"/features/{video_name}")[0:self.frames]
        print(X.columns)
        X = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X)
        print(X_tensor)
        
        y_raw = pd.read_csv(self.path + f"/labels/{video_name}")[0:self.frames]
        y = pd.Series(np.zeros(self.frames, dtype = int)-1)

        for behavior in self.behaviors:
            y[y_raw[behavior] == 1] = self.behaviors[behavior]

        if (y == -1).any():
            raise KeyError(terminal_colors.FAIL + f"{video_name} presents a behavior not specified in the behavior list: {self.behaviors}"+ terminal_colors.ENDC)

        y_tensor = torch.from_numpy(y.to_numpy())

        if self.transform:
            X_tensor = self.transform(X_tensor)
        if self.target_transform:
            y_tensor = self.target_transform(y_tensor)
        return X_tensor, y_tensor

