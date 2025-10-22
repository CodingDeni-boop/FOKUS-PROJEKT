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
from sklearn.model_selection import GridSearchCV


start=time.time()

################################## Load Data ###########################################################################

X_train, X_test, y_train, y_test = load_and_prepare_data()


################################ Basic Model ###########################################################################
"""
lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)
lr.fit(X_train, y_train)

evaluate_model(lr, X_train, y_train, X_test, y_test)
"""

################################### Feature Selection ##################################################################

#L1 REGULARIZATION (can recucde to 0)
def L1_regularization(lr, X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [0.01, 0.1, 1.0],  # Regularization strength
        "penalty": ["l1"],  # L1 regularization
        "solver": ["saga"]  # Solver for logistic regression / also possible; ovr
    }
    LR_L1 = GridSearchCV(
        lr,
        param_grid,
        cv=5 ,
        scoring ='f1_weighted'
    )

    # Fit GridSearchCV to training data
    LR_L1.fit(X_train, y_train)

    # Best L1 model after hyperparameter tuning
    best_L1_model = LR_L1.best_estimator_

    n_used_features = np.sum(best_L1_model.coef_ != 0, axis=0)
    print(f"Number of features used (L1): {n_used_features}")

    print("Best parameters (L1):\n", LR_L1.best_params_)

    evaluate_model(best_L1_model, X_train, y_train, X_test, y_test)

    feature_importance(best_L1_model, X_train, y_train, X_test, y_test)

# L2 REGULARIZATION (up to 0)
def L2_regularization(model, X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [1.0],  # Regularization strength
        "penalty": ["l2"],  # L2 regularization
        "solver": ["lbfgs"]  # Solver for logistic regression, also possible; saga, ovr
       # "multi_class": ["ovr"]
    }
    LR_L2 = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1_weighted'
    )

    LR_L2.fit(X_train, y_train)

    best_L2_model = LR_L2.best_estimator_
    n_used_features_L2 = np.sum(np.any(best_L2_model.coef_ != 0, axis=0))
    print(f"Number of features used (L2): {n_used_features_L2}")

    print("Best parameters (L2):\n", LR_L2.best_params_)

    evaluate_model(best_L2_model, X_train, y_train, X_test, y_test)

    feature_importance(best_L2_model, X_train, y_train, X_test, y_test)

########################################## Feature Importance ##########################################################

def feature_importance(model, X_train, y_train, X_test, y_test):
    coefs = model.coef_[0]  # Getting coefficients from the model
    coef_L1 = pd.DataFrame({
        'Feature': X_train.columns,  # Using feature names from X_train
        'Importance': np.abs(coefs)
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Important Features (L1):")
    print(coef_L1.head(10))  #
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=coef_L1.head(10))
    plt.title("Top 10 Feature Importances (L2)")
    plt.tight_layout()
    
    file_name = 'L2_features_top_10.png'
    file_path = os.path.join("output", file_name)
    plt.savefig(file_path, dpi=300)
    """

########################################### Model Traininig ############################################################

lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)

#L2_regularization(lr, X_train, y_train, X_test, y_test)
# Best C = 1.0

L1_regularization(lr, X_train, y_train, X_test, y_test)


end = time.time()
print("Time elapsed:", end-start)
