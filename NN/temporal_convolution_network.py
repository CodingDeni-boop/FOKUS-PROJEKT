import torch
import numpy as np
import pandas as pd
import os
from utilities import terminal_colors
from videodataset import VideoDataset
from sklearn.preprocessing import StandardScaler
import random as rd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import nn

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device = mps_device)
    print(terminal_colors.GREEN + "\nMPS device found\n" + terminal_colors.ENDC)
else:
    print (terminal_colors.WARNING + "\n[WARNING]: MPS device not found, computing on CPU\n" + terminal_colors.ENDC)

video_names = []
for i in range(1,22):
    if not i==5:
        video_names.append(f"OFT_left_{i}.csv")

video_path = "data"
video_names_training, video_names_test = train_test_split(video_names, test_size = 5, random_state = 42, shuffle = True)
behaviors = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}

scaler = StandardScaler()
for name in video_names_training:
    video = pd.read_csv(video_path+"/features/"+name)
    print(video)
    scaler.partial_fit(video)
print(terminal_colors.GREEN + "Partial fitted: " + terminal_colors.ENDC + f"{scaler}\n")

train_set = VideoDataset(path = video_path, 
                          video_names = video_names_training,
                          scaler = scaler,
                          frames = 18000,
                          behaviors = behaviors,
                          identity = "train dataset")

test_set = VideoDataset(path = video_path, 
                          video_names = video_names_test,
                          scaler = scaler,
                          frames = 18000,
                          behaviors = behaviors,
                          identity = "test dataset")

train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_data_loader = DataLoader(test_set, batch_size=1, shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(114*31, 128*31)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128*31, 64*5)
        self.batchnorm2 = nn.BatchNorm1d(64*5)
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(64*5, 4)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.relu1(x)
        
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)

        return x

network1 = NeuralNet().to(mps_device)
print(network1)

for batch, (X, y) in enumerate(train_data_loader):
    print(batch)