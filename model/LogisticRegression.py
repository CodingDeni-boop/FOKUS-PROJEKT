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

print("X shape", X.shape)


####################################### Train/Test Split ##########################################
X_train, X_test, y_train, y_test = video_train_test_split(
    X, y, test_videos=2)   ### takes seperate vidoes as test set

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

######################################### MISSING DATA ###########################################

X_train = X_train.dropna() # drop whole row instead
X_test = X_test.dropna()

#######################  SCALING (after splitting!!) ###############################################################
from sklearn.preprocessing import StandardScaler

# Get numeric features from the training set
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

sc = StandardScaler()
X_train[num_features] = sc.fit_transform(X_train[num_features])
X_test[num_features]  = sc.transform(X_test[num_features])

################################ Basic Model ###########################################################################
"""
lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)
lr.fit(X_train, y_train)

evaluate_model(lr, X_train, y_train, X_test, y_test)
"""

########################################## Feature Selection ################################################

L2_regularization(X_train, y_train, X_test, y_test)

