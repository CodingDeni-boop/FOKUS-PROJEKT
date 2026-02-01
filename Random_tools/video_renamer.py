import os
import shutil

for i in range(1,22):
    shutil.copy(src=f"collection/{i}/left.csv",dst=f"./../Neural_Networks/data/features/OFT_left_{i}.csv")

