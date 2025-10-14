import pandas as pd
import os

df1 = pd.read_csv("../Agreement_Comparison/data/Empty_Cage_Daniele_1_c1_labels.csv", index_col=0)
df2 = pd.read_csv("../Agreement_Comparison/data/Empty_Cage_Nataliia_1_c1_labels.csv", index_col=0)

data_dir = "labels/empty_cage/empty_cage_labels_"

video_numbers = [1, 2, 3, 4]

all_labels = {}

print(os.getcwd())

for folder in os.listdir("./empty_cage/"):
    for file in os.listdir(folder):
        print(file)





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


''' old
# Save the result to a new CSV
df_result.to_csv("result.csv", index = False)

# Convert columns into a single label column
df_result["label"] = df_result.idxmax(axis=1)  # this just takes column name with max value

df_result = pd.DataFrame(df_result["label"])

labels_vid1 = df_result

print(df_result)

#df_result.to_csv("label_results.csv", index=False, header = True)

'''



