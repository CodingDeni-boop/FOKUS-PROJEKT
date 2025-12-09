from pipeline_code.Video_generation import annotate_video_with_predictions
from pipeline_code.Shelf import Shelf
import pandas as pd
from pipeline_code.model_tools import predict_multiIndex


X_path = "./pipeline_saved_processes/dataframes/X_31.csv"
y_path = "./pipeline_saved_processes/dataframes/y_31.csv"
model_path = "./pipeline_saved_processes/models/HGB_actual.pkl"

X = pd.read_csv(X_path, index_col=["video_id", "frame"])
y = pd.read_csv(y_path, index_col=["video_id", "frame"])

X_train, X_test, y_train, y_test, HGB = Shelf.load(X = X, y = y, path = model_path)

y_pred = predict_multiIndex(HGB, X_test.index.get_level_values(0).unique(), X_test, smooth_prediction_frames = 10)

for name in y_pred.index.levels[0]:
    y_pred_video = y_pred.loc[name].squeeze().to_numpy()
    y_test_video = y_test.loc[name].squeeze().to_numpy()
    annotate_video_with_predictions(f"./pipeline_inputs/raw_videos/OFT_{name}.avi", y_pred_video, f"./pipeline_outputs/video_pred_vs_true/OFT_{name}.mp4", true_labels = y_test_video)

