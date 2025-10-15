import pandas as pd
import os

all_labels = {}

vids = []

vid_num = 1

for folder in os.listdir("./empty_cage/"):
    for file in os.listdir("./empty_cage/"+folder):
        vids.append(pd.read_csv("./empty_cage/"+folder+"/"+file, index_col=0))
    df_result = (vids[0] &  vids[1] & vids[2]).astype(int)

    df_result["label"] = df_result.idxmax(axis=1).astype("category")

    # Store in dictionary
    all_labels[f"video_{vid_num}"] = df_result[["label"]]
    vid_num += 1

print(all_labels)

combined_labels = pd.concat(all_labels.values(), keys=all_labels.keys(), names=['video_id', 'frame'])
print(combined_labels)

X_with_labels = combined_labels.copy()
save_path = "labels.csv"
X_with_labels.to_csv(save_path, index=True)


"""
for f in video_numbers:
    # Read both annotators' files
    df_1 = pd.read_csv(f"{data_dir}{vid_num}_c1_labels.csv", index_col=0)
    df_2 = pd.read_csv(f"{data_dir}{vid_num}_c1_labels.csv", index_col=0)

    # Calculate agreement
    df_result = (df_1 & df_2).astype(int)

    # Convert to single label column
    df_result["label"] = df_result.idxmax(axis=1)
    df_result = pd.DataFrame(df_result["label"])

    # Store in dictionary
    all_labels[f"video_{vid_num}"] = df_result

## what if sb says unsupportedrear, sb supported rear


 old
# Save the result to a new CSV
df_result.to_csv("result.csv", index = False)

# Convert columns into a single label column
df_result["label"] = df_result.idxmax(axis=1)  # this just takes column name with max value

df_result = pd.DataFrame(df_result["label"])

labels_vid1 = df_result

print(df_result)

#df_result.to_csv("label_results.csv", index=False, header = True)

"""



