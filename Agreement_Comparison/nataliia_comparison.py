import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

files = [
    "data/Empty_Cage_Daniele_1_c1_labels.csv",
    "data/Empty_Cage_Nata_1_c1_labels.csv",
    "data/Empty_Cage_Nataliia_1_c1_labels.csv",
]
names = ["Daniele", "Nata", "Nataliia"]

# Load directly, drop first column (index)
dfs = [pd.read_csv(f, header=0).iloc[:, 1:] for f in files]

# Convert to class indices
class_rows = [df.to_numpy().argmax(axis=1) for df in dfs]
mat = np.vstack(class_rows)
n_frames = mat.shape[1]

# Colors
cmap = ListedColormap(["#808080", "#d62728", "#2ca02c", "#ff7f0e"])

plt.figure(figsize=(16, 3.6))
plt.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

# Add separators between rows
plt.hlines(np.arange(0.5, mat.shape[0]-0.5, 1),
           -0.5, n_frames-0.5, colors="white", linewidth=6)

plt.yticks(range(len(names)), names)
ticks = np.linspace(0, n_frames-1, 11, dtype=int)
plt.xticks(ticks, ticks)

plt.xlabel("Frame")
plt.ylabel("Rater")

legend_patches = [
    Patch(facecolor="#808080", label="background"),
    Patch(facecolor="#d62728", label="supportedrear"),
    Patch(facecolor="#2ca02c", label="unsupportedrear"),
    Patch(facecolor="#ff7f0e", label="grooming"),
]
plt.legend(handles=legend_patches, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))

plt.tight_layout()
plt.show()