import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Nata_comparison import video_num, daniele, nataliia, nata, annotator_names, behaviors, annotators

print(f'\n********************************* Behavior Instances per Annotator; Video {video_num}***************************************')

def count_behavior_instances(df, behavior):
    """
    An instance starts when label changes from 0 to 1 and ends when it changes back to 0
    """
    behavior_series = df[behavior]
    changes = behavior_series.diff()
    instances = (changes > 0.5).sum()
    return instances

# Print behavior instances per annotator
print("\nBehavior instances per annotator:")
for name, df in zip(annotator_names, annotators):
    print(f"\n{name}:")
    for behavior in behaviors:
        if behavior == 'background':  # Skip background behavior
            continue
        instances = count_behavior_instances(df, behavior)
        print(f"{behavior}: {instances} instances")

# Visualization for behavior instances
summary_data = []
for name, df in zip(annotator_names, annotators):
    for behavior in behaviors:
        if behavior == 'background':  # Skip background behavior
            continue
        instances = count_behavior_instances(df, behavior)
        summary_data.append({
            'Annotator': name,
            'Behavior': behavior,
            'Instances': instances
        })

summary_df = pd.DataFrame(summary_data)

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Behavior', y='Instances', hue='Annotator', data=summary_df)
plt.title(f'Behavior Instances by Annotator (Video {video_num})')
plt.xlabel('Behavior Type')
plt.ylabel('Number of Instances')
plt.tight_layout()
plt.savefig(f'output/behavior_instances_{video_num}_plot.png', bbox_inches='tight', dpi=300)
plt.close()





