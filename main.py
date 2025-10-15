from Random_tools import tools
import os
import shutil
import pandas as pd
import numpy as np

print(os.getcwd())



for folder in os.listdir("./empty_cage/"):
    for file in os.listdir("./empty_cage/"+folder):
        print("./empty_cage/"+folder+"/"+file)
