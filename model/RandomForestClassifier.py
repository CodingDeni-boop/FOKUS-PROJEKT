
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

rf = RandomForestClassifier()
y = pd.read_csv("nataliia_labels.csv", index_col=["video_id","frame"])
X = pd.read_csv("features.csv", index_col=["video_id","frame"])
X = drop_non_analyzed_videos(X,y)
X, y = drop_last_frame(X,y)

# Missing Data
print("X shape:", X.shape)
print("X NA:", X.isnull().sum())
print("y NA:", y.isna().sum())

# Train/Test Split
X_train, X_test, y_train, y_test = video_train_test_split(
    X, y, test_videos=2)   ### takes seperate vidoes as test set

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced', 
    n_jobs=-1,
    verbose=True)

# Evaluate model
evaluate_model(rf, X_train, y_train, X_test, y_test)

######################################## FEATURE SELECTION #################################################

# Univariate Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Univariate FS
UVFS_Selector = SelectKBest(score_func=f_classif, k=10) # manually found best k (based on n features of the other FS techniques)
X_UVFS = UVFS_Selector.fit_transform(X_train, y_train)
X_UVFS_test = UVFS_Selector.transform(X_test)

# Scores
scores = UVFS_Selector.scores_  #ANOVA scores
pvalues = UVFS_Selector.pvalues_  #p-values for each feature


UVFS_selected_features = UVFS_Selector.get_feature_names_out(input_features=X_train.columns)
print("Selected Features (UVFS):\n", UVFS_selected_features)