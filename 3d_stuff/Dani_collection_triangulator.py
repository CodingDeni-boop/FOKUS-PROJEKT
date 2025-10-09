import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
from py3r.behaviour.tracking.tracking_collection import TrackingCollection as Collection
from py3r.behaviour.tracking.tracking_mv import TrackingMV as MultiView


options = opt(fps=30)

tracking_collection = Collection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection",options,MultiView)
tracking_collection.stereo_triangulate()

