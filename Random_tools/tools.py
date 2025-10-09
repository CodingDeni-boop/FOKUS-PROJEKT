import os
import shutil
import pandas as pd

def folder_for_collection(    
        input_path_left = "/Users/user/Desktop/helpful_functions/folder_me/Empty_Cage_Left",
        input_path_right = "/Users/user/Desktop/helpful_functions/folder_me/Empty_Cage_Right",
        calibration_path = "/Users/user/Desktop/helpful_functions/folder_me/calibration.json",
        output_path = "/Users/user/Desktop/helpful_functions/folder_me_output/collection",
        name_of_file_after_number_left = "_Empty_Cage_Left_Sync.csv",
        name_of_file_after_number_right = "_Empty_Cage_Right_Sync.csv"):

    for i in range(1,22):
        folder = output_path+f"/{i}_Empty_Cage_Sync"
        os.makedirs(folder)
        shutil.copy(calibration_path,folder)
        shutil.copy(input_path_left+f"/{i}"+name_of_file_after_number_left,folder)
        os.rename(folder+f"/{i}"+name_of_file_after_number_left,folder+"/left.csv")
        shutil.copy(input_path_right+f"/{i}"+name_of_file_after_number_right,folder)
        os.rename(folder+f"/{i}"+name_of_file_after_number_right,folder+"/right.csv")

def drop_secundary_columns(threshold=0.8,collection_folder_path="./oft_tracking/Empty_Cage/collection/"):

    for file in os.listdir(collection_folder_path):
        left = pd.read_csv(collection_folder_path+file+"/left.csv")
        right = pd.read_csv(collection_folder_path+file+"/right.csv")
        #I DROP EVERY COLUMN THAT HAS > threshold NA
        dropping = left.isna().sum()>left.shape[0]*threshold
        droppedleft = left.drop(columns=left.columns[dropping])
        dropping = left.isna().sum()>left.shape[0]*threshold
        droppedright = left.drop(columns=left.columns[dropping])
        droppedleft.to_csv(collection_folder_path+file+"/left.csv",index=False)
        droppedright.to_csv(collection_folder_path+file+"/right.csv",index=False)
        print(file,droppedleft.shape[1],droppedright.shape[1])



