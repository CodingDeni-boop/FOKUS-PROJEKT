from Random_tools import tools
import os
import shutil
import pandas as pd
import numpy as np

#tools.drop_secundary_columns("3d_stuff/oft_tracking/Empty_Cage/collection_before_preprocessing/","3d_stuff/oft_tracking/Empty_Cage/collection/",threshold=0.3)

collection_folder_path_src = "3d_stuff/oft_tracking/Empty_Cage/collection_after_preprocessing/"

for file in os.listdir(collection_folder_path_src):
    left = pd.read_csv(collection_folder_path_src+file+"/left.csv")
    right = pd.read_csv(collection_folder_path_src+file+"/right.csv")
    print(file,left.shape, right.shape)