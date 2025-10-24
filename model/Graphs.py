from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from model_tools import video_train_test_split
from model_tools import drop_non_analyzed_videos
from model_tools import drop_last_frame
from PerformanceEvaluation import evaluate_model
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from DataPreprocessing import preprocess_data
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
start=time.time()

X_train, X_test, y_train, y_test, pca = preprocess_data(
    features_file="features.csv",
    labels_file="nataliia_labels.csv",
)
print(X_train.shape, X_test.shape)

k = 20
selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(X_train, y_train)

# Get selected features
scores = selector.scores_
norm_scores = scores / np.max(scores)
feature_importance = pd.DataFrame({"column_name":pd.Series(X_train.columns),"importance":pd.Series(norm_scores)})
feature_importance=feature_importance.sort_values(by="importance",ignore_index=True,ascending=False)
print(feature_importance)

plt.figure(figsize=(40,30))
plt.title("Normalized Feature Importance with Univariate Feature Selection")
sns.barplot(data=feature_importance,y="column_name",x="importance")
plt.savefig("./Eval_output/Univariate_Feature_Selection.png")

corr_matrix = X_train.corr("spearman")
plt.figure(figsize=(40,30))
sns.heatmap(corr_matrix)
plt.savefig("./Eval_output/correalation_matrix.png")
end=time.time()

print("time elapsed:", f"{int((end-start)//3600)}h {int(((end-start)%3600)//60)}m {int((end-start)%60)}s")

