import py3r.behaviour as py3r
from py3r.behaviour.tracking.tracking import LoadOptions as opt
import json
from py3r.behaviour.tracking.tracking_mv import TrackingMV as mv


options = opt(fps=30)

with open('calibration.json') as f:
    calibration = json.load(f)

test = mv.from_yolo3r({"left_output":"oft_tracking/Empty_Cage/Empty_Cage_Left/1_Empty_Cage_Left_Sync.csv", "right_output":"oft_tracking/Empty_Cage/Empty_Cage_Right/1_Empty_Cage_Right_Sync.csv"},"1_Empty_Cage_multiview",
                      options,calibration)

test3d = test.stereo_triangulate()

test3dcol = py3r.TrackingCollection.from_yolo3r_folder("./oft_tracking/Empty_Cage/collection_test/",options,py3r.TrackingMV)

test3dcol_tri = test3dcol.stereo_triangulate()

test3dcol_tri.strip_column_names()

py3r.Tracking.rescale_by_known_distance

test3dcol_tri.rescale_by_known_distance("tl","br", 0.64, dims = ("x","y","z"))

test3dcol_tri.plot(trajectories=["nose"], static=["tl", "tr", "br", "bl"], lines = [("tl", "tr")], dims=("x", "y", "z"))


test3dcol_tri.save_3d_tracking_video_multi_view(out_path = "./Nata_Animation_Output")

# Animated Skeleton video
"""
test3dcol_tri.generate_video(
    video_name="vid1_tracking_animation.mp4",
    trajectories=["nose"],
    static=["tl", "tr", "br", "bl"],
    lines=[("tl", "tr"), ("tr", "br"), ("br", "bl"), ("bl", "tl")],  # Draw box outline
    skeleton_lines=[
        ("nose", "headcentre"),
        ("headcentre", "neck"),
        ("neck", "bodycentre"),
        ("bodycentre", "tailbase"),
        ("tailbase", "tailcentre"),
        ("tailcentre", "tailtip"),
        # Add hip/shoulder connections
        ("neck", "hipr"),
        ("neck", "hipl"),
        ("bodycentre", "hipr"),
        ("bodycentre", "hipl")
    ],
    dims=("x", "y", "z"),
    fps=30,  
    dpi=100,
    figsize=(10, 10)
)
"""