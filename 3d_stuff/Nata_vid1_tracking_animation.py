import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv
from sklearn import calibration

options = opt(fps=30)

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/",options,py3r.TrackingMV)

test3dcol_tri = test3dcol.stereo_triangulate()

test3dcol_tri.strip_column_names()

py3r.Tracking.rescale_by_known_distance

test3dcol_tri.rescale_by_known_distance("tl","br", 0.64, dims = ("x","y","z"))

# Skeleton Tracking Animation
test3dcol_tri.save_3d_tracking_video_multi_view(out_path = "./Nata_Animation_Output/vid_1_anim.gif", startframe = 1, endframe = 500, lines = [("tl","tr"),("tr","br"),("br","bl"),("bl","tl"),("nose", "neck"),("tailbase","tailcentre"),("tailcentre","tailtip")])

#ffmpeg not recognized!
#test3dcol_tri.save_3d_tracking_video_multi_view(out_path = "./Nata_Animation_Output/vid_1_anim.mp4", startframe = 1, endframe = 500, lines = [("nose", "neck")], writer ="ffmpeg")

