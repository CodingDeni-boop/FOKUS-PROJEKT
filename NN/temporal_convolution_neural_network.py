import torch
import numpy as np
import pandas as pd
import os
from utilities import terminal_colors as colors
from VideoDataSet import RandomizedDataset, SingleVideoDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import nn

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device = mps_device)
    print(colors.GREEN + "\nMPS device found\n" + colors.ENDC)
else:
    print (colors.WARNING + "\n[WARNING]: MPS device not found, computing on CPU\n" + colors.ENDC)

video_names = []
for i in range(1,22):
    if not i==5:
        video_names.append(f"OFT_left_{i}")

features_folder = "./data/rotated_videos"
labels_folder = "./data/labels"
video_names_train, video_names_test = train_test_split(video_names, test_size = 4, shuffle = True, random_state = 46)
behaviors = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}

train_set = RandomizedDataset(features_folder, labels_folder,  video_names_train, behaviors, 98, 31, 60, random_state = 42, identity = "TRAIN randomized dataset")

