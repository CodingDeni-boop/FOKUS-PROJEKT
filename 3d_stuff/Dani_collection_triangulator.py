import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
from py3r.behaviour.tracking.tracking_collection import TrackingCollection as Collection
from py3r.behaviour.tracking.tracking_mv import TrackingMV as MultiView


options = opt(fps=30)
tracking_collection = Collection.from_yolo3r_folder("./3d_stuff/oft_tracking/Empty_Cage/collection_before_preprocessing",options,MultiView)
triangulated_tracking_collection = tracking_collection.stereo_triangulate()
triangulated_tracking_collection.strip_column_names()
triangulated_tracking_collection.rescale_by_known_distance("tl","br", 0.64, dims = ("x","y","z"))
print(triangulated_tracking_collection[0].data)
triangulated_tracking_collection.plot(trajectories=["bodycentre"], static=["tl", "tr", "br", "bl"], lines = [("tl", "tr")], dims=("x", "y", "z"))

