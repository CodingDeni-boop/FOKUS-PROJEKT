import pandas as pd

df1 = pd.read_csv("./../Agreement_Comparison/data/Empty_Cage_Daniele_1_c1_labels.csv", index_col=0)
df2 = pd.read_csv("./../Agreement_Comparison/data/Empty_Cage_Nataliia_1_c1_labels.csv", index_col=0)

df_result = (df1 & df2).astype(int) #Checks where both Annotators agree

## what if sb says unsupportedrear, sb supported rear

# Save the result to a new CSV
df_result.to_csv("result.csv", index = False)

# Convert columns into a single label column
df_result["label"] = df_result.idxmax(axis=1)  # this just takes column name with max value

df_result.to_csv("label_results.csv", index=False, index_label=df_result.index.name)


