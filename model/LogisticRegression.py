from sklearn.ensemble import RandomForestClassifier
#from labels.deepethogram_vid_import import all_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
from FeatureSelection import *

y = pd.read_csv("nataliia_labels.csv", index_col=["video_id","frame"])
X = pd.read_csv("features.csv", index_col=["video_id","frame"])
X = drop_non_analyzed_videos(X,y)
X, y = drop_last_frame(X,y)

######################################### MISSING DATA ###########################################

# For time series data, forward fill then backward fill is most appropriate
X = X.fillna(method='ffill').fillna(method='bfill')
# If any NaN still remain (e.g., entire columns are NaN), fill with 0
X = X.fillna(0)

####################################### Train/Test Split ##########################################
X_train, X_test, y_train, y_test = video_train_test_split(
    X, y, test_videos=2)   ### takes seperate vidoes as test set

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

lr = LogisticRegression(random_state=42, class_weight='balanced' )

########################################## Feature Selection ################################################

L1_regularization(X_train, y_train)