from pipeline_code.generate_X import triangulate
import joblib as job
import time


# THE AIM OF THIS IS TO MAKE THIS A SINGLE FILE, WHICH USES OUR REPOSITORY AS A SORT OF LIBRARY. 
# THIS IS AN ATTEMPT TO MAKE ORDER


######      ######


fc = triangulate(
    collection_path = "./pipeline_inputs/collection",
    fps = 30,

    rescale_points = ("tr","tl"),
    rescale_distance = 0.64,
    filter_threshold = 0.9,
    construction_points = {"mid" : {"between_points" : ("tl", "tr", "bl", "br"), "mouse_or_oft" : "oft"}},
    smoothing = True,
    smoothing_mouse = 3,
    smoothing_oft = 20
    )
job.dump(fc, f"pipeline_saved_processes/Feature_Collection_{int(time.localtime()[2])}-{int(time.localtime()[1])}-{int(time.localtime()[0])}_{int(time.localtime()[3])}:{int(time.localtime()[4])}")

