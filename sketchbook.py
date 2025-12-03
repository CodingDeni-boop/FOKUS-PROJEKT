import pandas as pd
import numpy as np
import joblib as job
import os
import matplotlib.pyplot as plt
import seaborn as sns


def count_behavior_instances(df, behavior):
    """
    from Nata_Behavior_Num.py
    """
    behavior_series = df[behavior]
    changes = behavior_series.diff()
    instances = (changes > 0.5).sum()
    return instances

"""
if scatter_plot:
    colors = {'supportedrear': 'red', 'unsupportedrear': 'green', 'grooming': 'orange'}
    behaviors_to_plot = ['supportedrear', 'unsupportedrear', 'grooming']

    # Create 1x3 subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

        ax.set_xlabel('Person 1 Instance Count', fontsize=10)
        ax.set_ylabel('Person 2 Instance Count', fontsize=10)
        ax.set_title(f'{beh}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"\nTOTAL {beh}: Person 1 = {sum(person1_counts)}, Person 2 = {sum(person2_counts)}")

    plt.suptitle('Behaviour Instance Count Comparison: Person 1 vs Person 2', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pipelineoutput/scatterplot_all_behaviours.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatterplot saved to output/scatterplot_all_behaviours.png")
"""

