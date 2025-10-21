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
from FeatureSelection import L2_regularization
import time
from sklearn.linear_model import LogisticRegression
from Prepare_Data import load_and_prepare_data

start=time.time()

X_train, X_test, y_train, y_test = load_and_prepare_data()

"""

y = pd.read_csv("nataliia_labels.csv", index_col=["video_id","frame"])
X = pd.read_csv("features.csv", index_col=["video_id","frame"])
X = drop_non_analyzed_videos(X,y)
X, y = drop_last_frame(X,y)

print("X shape", X.shape)

######################################### MISSING DATA ###########################################

na_percentage = X.isna().mean()
columns_to_keep = na_percentage[na_percentage <= 0.1].index
columns_dropped = na_percentage[na_percentage > 0.1].index

print(f"Dropped {len(columns_dropped)} columns with >10% missing values:")
print(columns_dropped.tolist())
X = X[columns_to_keep]

valid_mask = X.notna().all(axis=1)
valid_X = X[valid_mask]
valid_y = y[valid_mask]

####################################### Train/Test Split ##########################################
X_train, X_test, y_train, y_test = video_train_test_split(
    valid_X, valid_y, test_videos=2)

# Convert labels to numpy arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

#######################  SCALING ####################
from sklearn.preprocessing import StandardScaler

num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

sc = StandardScaler()
X_train[num_features] = sc.fit_transform(X_train[num_features])
X_test[num_features] = sc.transform(X_test[num_features])
"""

################################ Basic Model ###########################################################################
"""
lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)
lr.fit(X_train, y_train)

evaluate_model(lr, X_train, y_train, X_test, y_test)
"""

########################################## Feature Selection ################################################

# Regularization

lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)
#lr_multi = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000, multi_class='multinomial')

L2_regularization(lr, X_train, y_train, X_test, y_test)
# Best C = 1.0



end = time.time()
print("Time elapsed:", end-start)
