import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import cohen_kappa_score

behavior = ['background', 'supportedrear', 'unsupportedrear', 'grooming']

comparisons = False

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

videos_dataframes = {}

for handle in videos:
    videos_dataframes[handle] = []
    for person in videos[handle]:
        videos_dataframes[handle].append(pd.read_csv("./data_final/"+handle+"_"+person+".csv").iloc[:,1:])

### PRINT COMPARISONS ###
if comparisons:
    for handle in videos_dataframes:
        plt.figure(figsize=(24, 3))
        timeline_data = []

        for df in videos_dataframes[handle]:
            cat_series = pd.Series('background', index=df.index)
            for cat in behavior[1:]:  # skip background
                cat_series[df[cat] == 1] = cat
            timeline_data.append(cat_series)

        colors = {'background': 'gray', 'supportedrear': 'red',
                'unsupportedrear': 'green', 'grooming': 'orange'}
        for i, (data, name) in enumerate(zip(timeline_data, videos[handle])):
            for cat in behavior:
                mask = data == cat
                if mask.any():
                    plt.scatter(data[mask].index, [i + 1] * mask.sum(),
                                label=cat if i == 0 else None,
                                alpha=0.6, c=colors[cat], marker='|', s=10000)

        plt.yticks(list(range(1, len(videos[handle]) + 1)), videos[handle])
        plt.xlabel('Frame Number')
        plt.title(f'Timeline of Behavior Classifications (Video {handle})')
        plt.legend([plt.Line2D([],[],color=colors[c],marker='.',linestyle='',markersize=15) for c in behavior],
            behavior,bbox_to_anchor=(1.005, 1))

        plt.tight_layout()
        plt.savefig(f'output/timeline_{handle}_plot.png', bbox_inches='tight', dpi=300)
        plt.close()

### PRINT CONFUSION MATRIX: PERSON1 ON COLUMN, PERSON2 ON ROW. FROM THIS WE CAN DO STATISTICS LIKE WE DO FOR MODEL. IT WILL BE GOOD FOR COMPARING. ###


for handle in videos:
    videos_dataframes[handle] = []
    for person in videos[handle]:
        one_hot_encoded = pd.read_csv("./data_final/"+handle+"_"+person+".csv").iloc[:,1:]
        categorical_df = pd.DataFrame()
        categorical_df["behaviour"] = one_hot_encoded.idxmax(axis=1)
        videos_dataframes[handle].append(categorical_df)

print(videos_dataframes)





