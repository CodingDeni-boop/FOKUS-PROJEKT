import pandas as pd
import shutil

for i in range(1,22):
    shutil.move(f'Empty Cage Videos/{i}/camera-1_0_synced.avi',"../tracker1/{i}_camera-1_0_synced.avi")
