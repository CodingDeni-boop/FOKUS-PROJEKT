from pipeline_code.generate_features import triangulate
from pipeline_code.generate_features import features
import joblib as job
import time


# THE AIM OF THIS IS TO MAKE THIS A SINGLE FILE, WHICH USES OUR REPOSITORY AS A SORT OF LIBRARY. 
# THIS IS AN ATTEMPT TO MAKE ORDER


######      ######


features_collection = triangulate(
    collection_path = "./pipeline_inputs/collection",
    fps = 30,

    rescale_points = ("tr","tl"),
    rescale_distance = 0.64,
    filter_threshold = 0.9,
    construction_points = {"mid" : {"between_points" : ("tl", "tr", "bl", "br"), "mouse_or_oft" : "oft"},},
    smoothing = True,
    smoothing_mouse = 3,
    smoothing_oft = 20
    )

a = features(features_collection, 
             distance = (("neck","earl"),
                        ("neck","earr"),
                        ("neck","bcl"),
                        ("neck","bcr"),
                        ("bcl","hipl"),
                        ("bcr","hipr"),
                        ("hipl","tailbase"),
                        ("hipr","tailbase"),
                        ("headcentre","neck"),
                        ("neck","bodycentre"),
                        ("bodycentre","tailbase"),
                        ("headcentre","earl"),
                        ("headcentre","earr"),
                        ("bodycentre","bcl"),
                        ("bodycentre","bcr"),
                        ("bodycentre","hipl"),
                        ("bodycentre","hipr")
                        )  
)

job.dump(fc, f"pipeline_saved_processes/Feature_Collection_{int(time.localtime()[2])}-{int(time.localtime()[1])}-{int(time.localtime()[0])}_{int(time.localtime()[3])}:{int(time.localtime()[4])}")

