import pandas as pd
import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV

options = opt(fps=30)

tracking_collection = TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options, TrackingMV)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("hipl","hipr", 0.05, dims = ("x","y","z"))

fc = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

# Azimuth
fc.azimuth('nose','neck').store()
fc.azimuth('neck', 'bodycentre').store()
fc.azimuth('headcentre', 'neck').store()

# Distance

pairs_of_points = pd.DataFrame({
    "point1": ["nose", "nose", "neck", "neck", "neck", "neck", "bcl", "bcr", "hipl", "hipr"],
    "point2": ["earl", "earr", "earl", "earr", "bcl", "bcr", "hipl", "hipr", "tailbase", "tailbase"]
})

for i in range(0,pairs_of_points.shape[0]):
    fc.distance_between(pairs_of_points.iloc[i,0],pairs_of_points.iloc[i,1],dims=("x","y","z")).store()
print(fc[0].data)

# Speed

videos = []
videos.append(file)

#Area of Mouse

embedding_dfs = {
    name: f.embedding_df(embedding_dict)
    for name, f in self.features_dict.items()
}
# Check all embeddings have the same columns
columns = next(iter(embedding_dfs.values())).columns
if not all(df.columns.equals(columns) for df in embedding_dfs.values()):
    raise ValueError("All embeddings must have the same columns")

# Concatenate with keys to create a MultiIndex
combined = pd.concat(embedding_dfs.values(), axis=0, keys=embedding_dfs.keys())
valid_mask = combined.notna().all(axis=1)
valid_combined = combined[valid_mask]