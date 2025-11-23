import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load labels to visualize class imbalance
labels_df = pd.read_csv("processed_labels.csv")
class_counts = labels_df['label'].value_counts().sort_index()

# Create class imbalance plot
plt.figure(figsize=(12, 6))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Class Imbalance Distribution', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("./Eval_output/class_imbalance.png")
print(f"\nClass distribution:\n{class_counts}")
print(f"\nClass imbalance plot saved to ./Eval_output/class_imbalance.png")
plt.show()

classes = ["Background", "Supported Rear", "Unsupported Rear", "Grooming"]

LR_f1 = [0.82, 0.76, 0.91, 0.88]
RF_f1 = [0.78, 0.74, 0.89, 0.85]
SVM_f1 = [0.80, 0.70, 0.93, 0.90]
HGB_f1 = [0.84, 0.79, 0.90, 0.87]

# Put into a list for easy handling
all_model_scores = [LR_f1, RF_f1, SVM_f1, HGB_f1]
model_names = ["LR", "RF", "SVM", "HGB"]

# Setup
x = np.arange(len(classes))  # positions for each class group
width = 0.2  # width of each bar

# Colorblind-friendly palette (works for deuteranopia and protanopia)
colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']  # Blue, Orange, Teal, Pink

plt.figure(figsize=(10, 6))

# Plot each model's bars
for i, model_scores in enumerate(all_model_scores):
    plt.bar(x + i * width, model_scores, width=width, label=model_names[i], color=colors[i])

# Labels and formatting
plt.xticks(x + width * 1.5, classes)
plt.ylabel("F1 Score")
plt.title("F1 Scores per Class for Each Model")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# Seaborn version
# Prepare data in long format for seaborn
data = []
for model_idx, model_name in enumerate(model_names):
    for class_idx, class_name in enumerate(classes):
        data.append({
            'Model': model_name,
            'Class': class_name,
            'F1 Score': all_model_scores[model_idx][class_idx]
        })

df = pd.DataFrame(data)

# Colorblind-friendly palette
palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='F1 Score', hue='Class', palette=palette)
plt.title("F1 Scores per Model for Each Class")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

