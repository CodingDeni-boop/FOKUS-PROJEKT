from Random_tools import tools
import os
import shutil
import pandas as pd
import numpy as np


collection_folder_path_src = "3d_stuff/oft_tracking/Empty_Cage/collection/"


for file in os.listdir(collection_folder_path_src):
    left = pd.read_csv(collection_folder_path_src+file+"/left.csv")
    right = pd.read_csv(collection_folder_path_src+file+"/right.csv")
    print(file,left.shape, right.shape)

#aa
