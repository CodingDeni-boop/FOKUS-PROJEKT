import pandas as pd
import numpy as np
import os

def drop_columns(collection_directory,):
    for file in os.listdir(collection_directory):
        if file.