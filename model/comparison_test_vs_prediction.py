import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PerformanceEvaluation import smooth_predictions

test = pd.read_csv("../model/svm_test.csv")
pred = pd.read_csv("../model/svm_prediction.csv")

label_map = {
    "background": 0,
    "supportedrear": 1,
    "unsupportedrear": 2,
    "grooming": 3,
}

y_true = test.iloc[:, 0].map(label_map).to_numpy()
y_pred = pred.iloc[:, 0].map(label_map).to_numpy()

# Apply smoothing to remove outliers below x frames
y_pred_smooth = smooth_predictions(y_pred, min_frames=20)

# Create smoothed prediction DataFrame for instance counting
inv_label_map = {v: k for k, v in label_map.items()}
pred_smoothed = pd.DataFrame(pd.Series(y_pred_smooth).map(inv_label_map), columns=pred.columns)

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
conf_df = pd.crosstab(
    pd.Series(y_true).map(inv_label_map),
    pd.Series(y_pred).map(inv_label_map),
    rownames=["True"],
    colnames=["Predicted"],
    dropna=False,
)

print("\nConfusion matrix:")
print(conf_df)

# Behaviour Frames counts
print("\n=== Behaviour Frame Counts ===")
test_counts = test.iloc[:, 0].value_counts().sort_index()
pred_counts = pred_smoothed.iloc[:, 0].value_counts().sort_index()

comparison_df = pd.DataFrame({
    "Test": test_counts,
    "Prediction": pred_counts,
    "Difference": pred_counts - test_counts
}).fillna(0).astype(int)

print(comparison_df)

# Behavior Instances

def count_behavior_instances(df, behavior_name):
    """
    Count instances of a specific behavior in the label column.
    An instance starts when label changes to the behavior and ends when it changes to something else.
    """
    label_series = df.iloc[:, 0]
    is_behavior = (label_series == behavior_name).astype(int)
    changes = is_behavior.diff()
    instances = (changes > 0.5).sum()
    return instances

behaviors = ["supportedrear", "unsupportedrear", "grooming"]

print("\n=== Behaviour Instance Counts ===")

print("\nWithout smoothing:")
for behavior in behaviors:
    count_test = count_behavior_instances(test, behavior)
    count_pred = count_behavior_instances(pred, behavior)
    difference = count_pred - count_test

    print(f"\n{behavior}:")
    print(f"  Test: {count_test} instances")
    print(f"  Prediction: {count_pred} instances")
    print(f"  Difference: {difference}")

print("\nWith smoothing:")
instance_data = []
for behavior in behaviors:
    count_test = count_behavior_instances(test, behavior)
    count_pred = count_behavior_instances(pred_smoothed, behavior)
    difference = count_pred - count_test

    print(f"\n{behavior}:")
    print(f"  Test: {count_test} instances")
    print(f"  Prediction: {count_pred} instances")
    print(f"  Difference: {difference}")

    instance_data.append({
        'Behaviour': behavior,
        'Test': count_test,
        'Prediction': count_pred
    })

# Create bar plot for behavior instance counts
instance_df = pd.DataFrame(instance_data)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(behaviors))
width = 0.35

bars1 = ax.bar(x - width/2, instance_df['Test'], width, label='Test', color='#0173B2')
bars2 = ax.bar(x + width/2, instance_df['Prediction'], width, label='Prediction (Smoothed)', color='#029E73')

ax.set_xlabel('Behaviour Class', fontsize=12)
ax.set_ylabel('Number of Instances', fontsize=12)
ax.set_title('Behaviour Instance Count per Class', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(behaviors, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("./Eval_output/behaviour_instance_count.png")
print(f"\nBehaviour instance count plot saved to ./Eval_output/behaviour_instance_count.png")
plt.show()