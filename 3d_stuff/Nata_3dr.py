import py3r.behaviour as bv

# load a folder of tracking files as a collection
tc = bv.TrackingCollection.from_dlc_folder('path/to/folder')

# generate features for all animals
fc = bv.FeaturesCollection.from_tracking_collection(tc)
