import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

test = pd.read_csv("test.csv")
pred = pd.read_csv("prediction.csv")

label_map = {
    "background": 0,
    "supportedrear": 1,
    "unsupportedrear": 2,
    "grooming": 3,
}

y_true = test["0"].map(label_map).to_numpy()
y_pred = pred["0"].map(label_map).to_numpy()

mat = np.vstack([y_true, y_pred])
names = ["Test", "Prediction"]
n_frames = mat.shape[1]

cmap = ListedColormap(["#808080", "#d62728", "#2ca02c", "#ff7f0e"])

plt.figure(figsize=(16, 3.6))
plt.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

plt.hlines(np.arange(0.5, mat.shape[0]-0.5, 1),
           -0.5, n_frames-0.5, colors="white", linewidth=6)

plt.yticks(range(len(names)), names)

ticks = np.linspace(0, n_frames-1, 11, dtype=int)
plt.xticks(ticks, ticks)

plt.xlabel("Frame")
plt.ylabel("Source")

legend_patches = [
    Patch(facecolor="#808080", label="background"),
    Patch(facecolor="#d62728", label="supportedrear"),
    Patch(facecolor="#2ca02c", label="unsupportedrear"),
    Patch(facecolor="#ff7f0e", label="grooming"),
]
plt.legend(handles=legend_patches, ncols=4, loc="upper center",
           bbox_to_anchor=(0.5, 1.15))

plt.tight_layout()
plt.show()

# Accuracy
accuracy = (y_true == y_pred).mean()
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
inv_label_map = {v: k for k, v in label_map.items()}

conf_df = pd.crosstab(
    pd.Series(y_true).map(inv_label_map),
    pd.Series(y_pred).map(inv_label_map),
    rownames=["True"],
    colnames=["Predicted"],
    dropna=False,
)

print("\nConfusion matrix:")
print(conf_df)