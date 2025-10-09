import os
import shutil


input_path_left = "/Users/user/Desktop/helpful_functions/folder_me/Empty_Cage_Left"
input_path_right = "/Users/user/Desktop/helpful_functions/folder_me/Empty_Cage_Right"
calibration_path = "/Users/user/Desktop/helpful_functions/folder_me/calibration.json"
output_path = "/Users/user/Desktop/helpful_functions/folder_me_output/collection"
name_of_file_after_number_left = "_Empty_Cage_Left_Sync.csv"
name_of_file_after_number_right = "_Empty_Cage_Right_Sync.csv"


for i in range(1,22):
    folder = output_path+f"/{i}_Empty_Cage_Sync"
    os.makedirs(folder)
    shutil.copy(calibration_path,folder)
    shutil.copy(input_path_left+f"/{i}"+name_of_file_after_number_left,folder)
    os.rename(folder+f"/{i}"+name_of_file_after_number_left,folder+"/left.csv")
    shutil.copy(input_path_right+f"/{i}"+name_of_file_after_number_right,folder)
    os.rename(folder+f"/{i}"+name_of_file_after_number_right,folder+"/right.csv")

