import pandas as pd
from natsort import natsorted
import os

def labels(labels_path : str,
           output_path : str
           ):

    dictionary = {}
    
    for name in natsorted(os.listdir(labels_path)):
        df = pd.read_csv(labels_path + "/" + name, index_col=[0])
        dictionary[name.replace(".csv","")] = df.idxmax(axis = 1)

    y = pd.concat(dictionary.values(), keys=dictionary.keys(), names=["video_id", "frame"])
    y.to_csv(output_path)
    print("!labels_saved!")

    return y