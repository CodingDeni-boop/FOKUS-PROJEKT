import pandas as pd
import os

all_labels = {}

vid_num = 1

for file in sorted(os.listdir("./nataliia_empty_cage/")):
    df = pd.read_csv("./nataliia_empty_cage/" + file, index_col=0)
    df["label"] = df.idxmax(axis=1)
    all_labels[f"{vid_num}_Empty_Cage_Sync"] = df[["label"]]
    vid_num += 1

print(all_labels)

combined_labels = pd.concat(all_labels.values(), keys=all_labels.keys(), names=['video_id', 'frame'])
print(combined_labels)

X_with_labels = combined_labels.copy()
save_path = "../model/nataliia_labels.csv"
X_with_labels.to_csv(save_path, index=True)