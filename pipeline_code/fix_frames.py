import pandas as pd

def drop_non_analyzed_videos(X : pd.DataFrame,y : pd.DataFrame):
    y_index = y.index.get_level_values("video_id").unique()
    return X.loc[y_index]

def drop_last_frame(X : pd.DataFrame,y : pd.DataFrame):
    
    X_index = X.index.get_level_values("video_id").unique()
    y_index = y.index.get_level_values("video_id").unique()
    if not (X_index.equals(y_index)):
        raise ValueError("X index name doesn't match y index name")
    index = X_index
    while X.shape[0]!=y.shape[0]:
        for video_name in index:
            if y.loc[video_name].shape[0] == X.loc[video_name].shape[0]:
                continue

            elif y.loc[video_name].shape[0] > X.loc[video_name].shape[0]:
                difference = y.loc[video_name].shape[0] - X.loc[video_name].shape[0]
                y = y.drop((video_name, y.loc[video_name].index[-1]))
                print(f"video '{video_name}' has {difference} too many frames in y: dropped 1")

            elif y.loc[video_name].shape[0] < X.loc[video_name].shape[0]:
                difference = X.loc[video_name].shape[0] - y.loc[video_name].shape[0]
                X = X.drop((video_name, X.loc[video_name].index[-1]))
                print(f"video '{video_name}' has {difference} too many frames in X: dropped 1")

    return X, y

