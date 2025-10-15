from Random_tools import tools
import os
import shutil
import pandas as pd
import numpy as np
import csv



print(os.getcwd())

all_labels = {}

vids = []

vid_num = 1

for folder in os.listdir("./empty_cage/"):
    for file in os.listdir("./empty_cage/"+folder):
        vids.append(pd.read_csv("./empty_cage/"+folder+"/"+file))

print(vids)