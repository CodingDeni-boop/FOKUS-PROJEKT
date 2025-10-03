from Nata_comparison import *


def count_behavior_instances(df):
    # Convert DataFrame to behavior series
    cat_series = pd.Series('background', index=df.index)
    for cat in behaviors:
        cat_series[df[cat] == 1] = cat

    # Count behavior instances (when behavior changes from background to specific behavior)
    behavior_counts = {}
    current_behavior = 'background'
    behavior_instances = {b: 0 for b in behaviors}

    for behavior in cat_series:
        if behavior != current_behavior:
            if behavior != 'background':
                behavior_instances[behavior] += 1
            current_behavior = behavior

    return behavior_instances


print("\nBehavior instances per annotator:")
for name, df in zip(annotator_names, annotators):
    print(f"\n{name}:")
    behavior_instances = count_behavior_instances(df)
    for behavior, count in behavior_instances.items():
        print(f"{behavior}: {count} instances")

