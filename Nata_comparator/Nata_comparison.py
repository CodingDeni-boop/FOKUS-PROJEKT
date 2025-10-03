import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import cohen_kappa_score

video_num = input("Enter Video Number: ")

daniele = pd.read_csv(f'data/Empty_Cage_Daniele_{video_num}_c1_labels.csv', index_col=0)
nataliia = pd.read_csv(f'data/Empty_Cage_Nataliia_{video_num}_c1_labels.csv', index_col=0)
nata = pd.read_csv(f'data/Empty_Cage_Nata_{video_num}_c1_labels.csv', index_col=0)

print(f'\n****************************************** Video {video_num} ***********************************************')

########################################################################################################################
behavior = ['background', 'supportedrear', 'unsupportedrear', 'grooming']
annotators = [daniele, nataliia, nata]
annotator_names = ['Daniele', 'Nataliia', 'Nata']

# Calculate agreement per behavior label
def calculate_behavior_agreement(df1, df2, behavior):
    # Calculate agreement only for the specific behavior
    agreement = np.mean((df1[behavior] == df2[behavior])) * 100
    return agreement

################################################# TIMELINE PLOT ########################################################
plt.figure(figsize=(15, 3))
#plt.grid(True, which='major', axis='x', linestyle='-', alpha=0.3)

timeline_data = []

for df in annotators:
    cat_series = pd.Series('background', index=df.index)
    for cat in behavior[1:]:  # skip background
        cat_series[df[cat] == 1] = cat
    timeline_data.append(cat_series)

colors = {'background': 'gray', 'supportedrear': 'blue',
          'unsupportedrear': 'red', 'grooming': 'green'}

for i, (data, name) in enumerate(zip(timeline_data, annotator_names)):
    for cat in behavior:
        mask = data == cat
        if mask.any():
            plt.scatter(data[mask].index, [i + 1] * mask.sum(),
                        label=cat if i == 0 else None,
                        alpha=0.6, c=colors[cat], marker='|', s=500)

plt.yticks([1, 2, 3], ['Daniele', 'Nataliia', 'Nata'])
plt.xlabel('Frame Number')
plt.title(f'Timeline of Behavior Classifications (Video {video_num})')
plt.legend(bbox_to_anchor=(1.005, 1), loc='upper left')
#plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0)

plt.tight_layout()
plt.savefig(f'output/timeline_{video_num}_plot.png', bbox_inches='tight', dpi=300)

########################################### OVERALL AGREEMENT HEATMAP ##################################################

plt.figure(figsize=(10, 8))
agreement_matrix = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        if i != j:
            agreement = calculate_behavior_agreement(annotators[i], annotators[j], behavior)
            agreement_matrix[i][j] = agreement
        else:
            agreement_matrix[i][j] = 100.0

sns.heatmap(agreement_matrix,
            annot=True,
            fmt='.1f',
            xticklabels=annotator_names,
            yticklabels=annotator_names,
            cmap='PuBu',
            vmin=95, vmax=100)
plt.title(f'Agreement Between Annotators (%), Video {video_num}')

plt.tight_layout()
plt.savefig(f'output/overall_agreement_heatmap_{video_num}_plot.png', bbox_inches='tight', dpi=300)

# Labeled frames per Annotator
print("\nBehavior counts per annotator:")
for name, df in zip(annotator_names, annotators):
    print(f"\n{name}:")
    for cat in behavior:
        if cat == 'background':
            continue
        count = df[cat].sum()
        print(f"{cat}: {count} frames")

####################################### AGREEMENT HEATMAP FOR EACH BEHAVIOR ############################################

# Behaviors (without background)
behaviors = ['supportedrear', 'unsupportedrear', 'grooming']

#Matrices for each behavior
behavior_agreements = {}
for behavior in behaviors:
    agreement_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                agreement = calculate_behavior_agreement(annotators[i], annotators[j], behavior)
                agreement_matrix[i][j] = agreement
            else:
                agreement_matrix[i][j] = 100.0
    behavior_agreements[behavior] = agreement_matrix

#Heatmaps for each behavior
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Agreement Between Annotators per Behavior (%), Video {video_num}', fontsize=16, y=1.05)

for idx, (behavior, matrix) in enumerate(behavior_agreements.items()):
    sns.heatmap(matrix,
                annot=True,
                fmt='.1f',
                xticklabels=annotator_names,
                yticklabels=annotator_names,
                cmap='PuBu',
                vmin=95, vmax=100,
                ax=axes[idx])
    axes[idx].set_title(f'{behavior.capitalize()}')

plt.tight_layout()
plt.savefig(f'output/behavior_agreement_{video_num}_heatmaps.png', bbox_inches='tight', dpi=300)


#Print average agreement per behavior
print("\nAverage agreement per behavior:")
for behavior in behaviors:
    matrix = behavior_agreements[behavior]
    # Calculate mean of non-zero elements
    mean_agreement = matrix[matrix != 0].mean()
    print(f"{behavior}: {mean_agreement:.1f}%")

############## Cohens Kappa ##################################

print("\nCohen's Kappa Score:")

cohen_kappa_score(daniele[behavior], nataliia[behavior])
print("Daniele - Nataliia",cohen_kappa_score(daniele[behavior], nataliia[behavior]))
cohen_kappa_score(daniele[behavior], nata[behavior])
print("Daniele - Nata", cohen_kappa_score(daniele[behavior], nata[behavior]))
cohen_kappa_score(nataliia[behavior], nata[behavior])
print("Nataliia - Nata", cohen_kappa_score(nataliia[behavior], nata[behavior]))