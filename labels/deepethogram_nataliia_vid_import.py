import pandas as pd
import os
import re

input_dir = "./nataliia_empty_cage/"
output_dir = "../model/"
os.makedirs(output_dir, exist_ok=True)

def extract_number(filename):
    """Extracts the numeric ID from filename like 'Empty_Cage_Nataliia_10_c1_labels.csv'."""
    match = re.search(r'_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

all_labels = {}

# Sort files numerically by number inside filename
files = sorted(
    [f for f in os.listdir(input_dir) if f.endswith(".csv")],
    key=extract_number
)

for file in files:
    df = pd.read_csv(os.path.join(input_dir, file), index_col=0)
    df["label"] = df.idxmax(axis=1)

    # Extract numeric part for naming
    match = re.search(r'_(\d+)_', file)
    vid_num = match.group(1) if match else "unknown"

    # Desired standardized name
    video_name = f"{vid_num}_Empty_Cage_Sync"
    all_labels[video_name] = df[["label"]]

    print(f"{file} → {video_name}")

# Combine all labeled DataFrames
combined_labels = pd.concat(all_labels.values(), keys=all_labels.keys(), names=["video_id", "frame"])

# Save output
save_path = os.path.join(output_dir, "nataliia_labels.csv")
combined_labels.to_csv(save_path, index=True)

print(f"Combined labels saved to: {save_path}")
