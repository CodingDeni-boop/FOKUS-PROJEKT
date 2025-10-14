from Features import *

############################################## my code ###########################################################

first_F = next(iter(fc.features_dict.values()))
cols = first_F.tracking.data.columns

for col in cols:
    if col.endswith(".x"):
        p = col[:-2]
        fc.speed(p, dims=("x","y","z")).store()

print(fc[0].data)