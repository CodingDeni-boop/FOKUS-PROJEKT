import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score


behaviour = ['background', 'supportedrear', 'unsupportedrear', 'grooming']

comparisons = False
agreement_confusion_matrix = True
agreement_confusion_matrix_final = True
scatter_plot = True


videos = { "1" : ["Nata", "Nataliia"],
           "2" : ["Nata", "Nataliia"],
           "3" : ["Nata", "Nataliia"],
           "4" : ["Nata", "Nataliia"],
           "5" : ["Nata", "Nataliia"],
           "6" : ["Radu", "Nataliia"],
           "7" : ["Radu", "Nataliia"],
           "8" : ["Radu", "Nataliia"],
           "9" : ["Radu", "Nataliia"],
           "10" : ["Radu", "Nataliia"],
           "11" : ["Radu", "Nataliia"],
           "12" : ["Radu", "Nataliia"],
           "13" : ["Radu", "Nataliia"],
           "14" : ["Radu", "Nataliia"],
           "15" : ["Radu", "Daniele"],
           "16" : ["Radu", "Nata"],
           "17" : ["Nata", "Nataliia"],
           "18" : ["Radu", "Daniele"],
           "19" : ["Radu", "Daniele"],
           "20" : ["Daniele", "Nataliia"],
           "21" : ["Daniele", "Nataliia"]
        }

videos_final = {"1" : ["Nataliia"]}

videos_dataframes = {}
videos_final_dataframes = {}

## Reading file ##

for handle in videos:
    videos_dataframes[handle] = []
    for person in videos[handle]:
        videos_dataframes[handle].append(pd.read_csv("./data_final/"+handle+"_"+person+".csv").iloc[:,1:])

for handle in videos_final:
    videos_final_dataframes[handle] = []
    for person in videos_final[handle]:
        videos_final_dataframes[handle].append(pd.read_csv("./data_final_final/"+handle+"_final_"+person+".csv").iloc[:,1:])

### PRINT COMPARISONS ###
if comparisons:
    for handle in videos_dataframes:
        plt.figure(figsize=(24, 3))
        timeline_data = []

        for df in videos_dataframes[handle]:
            cat_series = pd.Series('background', index=df.index)
            for cat in behaviour[1:]:  # skip background
                cat_series[df[cat] == 1] = cat
            timeline_data.append(cat_series)

        colors = {'background': 'gray', 'supportedrear': 'red',
                'unsupportedrear': 'green', 'grooming': 'orange'}
        for i, (data, name) in enumerate(zip(timeline_data, videos[handle])):
            for cat in behaviour:
                mask = data == cat
                if mask.any():
                    plt.scatter(data[mask].index, [i + 1] * mask.sum(),
                                label=cat if i == 0 else None,
                                alpha=0.6, c=colors[cat], marker='|', s=10000)

        plt.yticks(list(range(1, len(videos[handle]) + 1)), videos[handle])
        plt.xlabel('Frame Number')
        plt.title(f'Timeline of behaviour Classifications (Video {handle})')
        plt.legend([plt.Line2D([],[],color=colors[c],marker='.',linestyle='',markersize=15) for c in behaviour],
            behaviour,bbox_to_anchor=(1.005, 1))

        plt.tight_layout()
        plt.savefig(f'output/timeline_{handle}_plot.png', bbox_inches='tight', dpi=300)
        plt.close()


## NOT ONE HOT ENCODED ##

for handle in videos:
    videos_dataframes[handle] = []
    for person in videos[handle]:
        one_hot_encoded = pd.read_csv("./data_final/"+handle+"_"+person+".csv").iloc[:,1:]
        categorical_df = pd.DataFrame()
        categorical_df["behaviour"] = one_hot_encoded.idxmax(axis=1)
        videos_dataframes[handle].append(categorical_df)

for handle in videos_final:
    videos_final_dataframes[handle] = []
    for person in videos_final[handle]:
        one_hot_encoded = pd.read_csv("./data_final_final/"+handle+"_final_"+person+".csv").iloc[:,1:]
        categorical_df = pd.DataFrame()
        categorical_df["behaviour"] = one_hot_encoded.idxmax(axis=1)
        videos_final_dataframes[handle].append(categorical_df)

### PRINT CONFUSION MATRIX: PERSON1 ON COLUMN, PERSON2 ON ROW. FROM THIS WE CAN DO STATISTICS LIKE WE DO FOR MODEL. IT WILL BE GOOD FOR COMPARING. ###

if agreement_confusion_matrix:

    chunky1 = pd.Series()
    chunky2 = pd.Series()

    for handle in videos_dataframes:
        chunky1 = pd.concat([chunky1, videos_dataframes[handle][0]["behaviour"]],axis = 0,ignore_index=True)
        chunky2 = pd.concat([chunky2, videos_dataframes[handle][1]["behaviour"]],axis = 0,ignore_index=True)
    
    cm = confusion_matrix(chunky1, chunky2, labels = behaviour)
    print(cm)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = np.round(cmn, 2)
    print(cmn)

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cmn,
        annot=True,           # Show numbers in cells
        cmap='Blues',         # Color scheme
        cbar_kws={'label': 'Count'},
        xticklabels=behaviour,
        yticklabels=behaviour
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel("person_1", fontsize=12)
    plt.xlabel("person_2", fontsize=12)
    plt.tight_layout()
    plt.savefig('output/person_1_vs_person_2_confusion_matrix.png', dpi=300, bbox_inches='tight')


if agreement_confusion_matrix_final:

    chunky1 = pd.Series()
    chunky2 = pd.Series()

    for handle in videos_final_dataframes:
        chunky1 = pd.concat([chunky1, videos_final_dataframes[handle][0]["behaviour"]],axis = 0,ignore_index=True)
        chunky2 = pd.concat([chunky2, videos_dataframes[handle][1]["behaviour"]],axis = 0,ignore_index=True)
    
    cm = confusion_matrix(chunky1, chunky2, labels = behaviour)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    # Confusion Matrix Plot
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in cells
        fmt='d',              # Format as integers
        cmap='Blues',         # Color scheme
        cbar_kws={'label': 'Count'},
        xticklabels=behaviour,
        yticklabels=behaviour
    )
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
    plt.ylabel("person_1", fontsize=15)
    plt.xlabel("person_2", fontsize=15)
    plt.tight_layout()
    plt.savefig('output/person_1_vs_person_2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(classification_report(chunky1, chunky2, labels = behaviour))


# Scatter plot comparing person 1 and person 2 - one plot per behavior

def count_behavior_instances(df, behavior):
    """
    from Nata_Behavior_Num.py
    """
    behavior_series = df[behavior]
    changes = behavior_series.diff()
    instances = (changes > 0.5).sum()
    return instances

if scatter_plot:
    colors = {'supportedrear': 'red', 'unsupportedrear': 'green', 'grooming': 'orange'}
    behaviors_to_plot = ['supportedrear', 'unsupportedrear', 'grooming']

    # Create 1x3 subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create one subplot for each behavior (excluding background)
    for idx, beh in enumerate(behaviors_to_plot):
        person1_counts = []
        person2_counts = []

        print(f"\n{'='*60}")
        print(f"Behavior: {beh}")
        print(f"{'='*60}")

        for handle in videos:
            # Read one-hot encoded data for counting instances
            p1_df = pd.read_csv("./data_final/"+handle+"_"+videos[handle][0]+".csv").iloc[:,1:]
            p2_df = pd.read_csv("./data_final/"+handle+"_"+videos[handle][1]+".csv").iloc[:,1:]

            # Count instances of this behavior for each person in this video
            p1_count = count_behavior_instances(p1_df, beh)
            p2_count = count_behavior_instances(p2_df, beh)

            person1_counts.append(p1_count)
            person2_counts.append(p2_count)

            print(f"Video {handle}: {videos[handle][0]}={p1_count}, {videos[handle][1]}={p2_count}")

        # Create scatter plot for this behavior in subplot
        ax = axes[idx]
        ax.scatter(person1_counts, person2_counts, c=colors[beh], alpha=0.6, s=100)

        # Add diagonal line for perfect agreement
        if person1_counts and person2_counts:
            max_val = max(max(person1_counts), max(person2_counts))
            if max_val > 0:
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect Agreement')

        ax.set_xlabel('Person 1 Instance Count', fontsize=14)
        ax.set_ylabel('Person 2 Instance Count', fontsize=14)
        ax.set_title(f'{beh}', fontsize=17, fontweight='bold', pad=10)
        ax.legend(fontsize=10.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        print(f"\nTOTAL {beh}: Person 1 = {sum(person1_counts)}, Person 2 = {sum(person2_counts)}")

    plt.suptitle('Behaviour Instance Count Comparison: Person 1 vs Person 2', fontsize=20, fontweight='bold', y=1)
    plt.tight_layout()
    plt.savefig('output/scatterplot_all_behaviours.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatterplot saved to output/scatterplot_all_behaviours.png")



