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
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from DataPreprocessing import preprocess_data
from FeatureSelection import apply_pca, apply_uvfs


start=time.time()



################################ Basic Model ###########################################################################
"""
lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)
lr.fit(X_train, y_train)

evaluate_model(lr, X_train, y_train, X_test, y_test)
"""

################################### Feature Selection ##################################################################

# Univariate Feature Selection
def univariateFS(lr, X_train, y_train, X_test, y_test, k):
    UVFS_Selector = SelectKBest(score_func=f_classif,
                                k=k)
    X_UVFS = UVFS_Selector.fit_transform(X_train, y_train)
    X_UVFS_test = UVFS_Selector.transform(X_test)

    # Scores
    scores = UVFS_Selector.scores_  # ANOVA scores
    pvalues = UVFS_Selector.pvalues_  # p-values for each feature

    UVFS_selected_features = UVFS_Selector.get_feature_names_out(input_features=X_train.columns)
    print("Selected Features (UVFS):\n", UVFS_selected_features)

    LR_UVFS = LogisticRegression(random_state=10, class_weight='balanced')
    LR_UVFS.fit(X_UVFS, y_train)

    evaluate_model(LR_UVFS, X_UVFS, y_train, X_UVFS_test, y_test)


#L1 REGULARIZATION (can reduce to 0)
def L1_regularization(lr, X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [1.0],  # Regularization strength
        "penalty": ["l1"],  # L1 regularization
        "solver": ["saga"]  # Solver for logistic regression / also possible; ovr
    }
    LR_L1 = GridSearchCV(
        lr,
        param_grid,
        cv=5 ,
        scoring ='f1_weighted',
        n_jobs=-1
    )

    # Fit GridSearchCV to training data
    LR_L1.fit(X_train, y_train)

    # Best L1 model after hyperparameter tuning
    best_L1_model = LR_L1.best_estimator_

    n_used_features = np.sum(best_L1_model.coef_ != 0, axis=0)
    print(f"Number of features used (L1): {n_used_features}")

    print("Best parameters (L1):\n", LR_L1.best_params_)

    evaluate_model(best_L1_model, X_train, y_train, X_test, y_test)

    feature_importance(best_L1_model, X_train, y_train, X_test, y_test, pca=pca, original_features=original_features)


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
        scoring='f1_weighted',
        n_jobs=-1
    )

    LR_L2.fit(X_train, y_train)

    best_L2_model = LR_L2.best_estimator_
    n_used_features_L2 = np.sum(np.any(best_L2_model.coef_ != 0, axis=0))
    print(f"Number of features used (L2): {n_used_features_L2}")

    print("Best parameters (L2):\n", LR_L2.best_params_)

    evaluate_model(best_L2_model, X_train, y_train, X_test, y_test)

    feature_importance(best_L2_model, X_train, y_train, X_test, y_test, pca=pca, original_features=original_features)


########################################## Feature Importance ##########################################################
"""
def feature_importance(model, X_train, y_train, X_test, y_test):
    coefs = model.coef_[0]  # Getting coefficients from the model
    coef_L1 = pd.DataFrame({
        'Feature': X_train.columns,  # Using feature names from X_train
        'Importance': np.abs(coefs)
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Important Features (L1):")
    print(coef_L1.head(10))  #
"""
def feature_importance(model, X_train, y_train, X_test, y_test, pca=None, original_features=None):
    """
    Print feature importances for logistic regression models.
    If PCA was used, map the coefficients back to the original feature space.
    """

    coefs = model.coef_[0]  # shape: (n_components,) if PCA used

    # === Case 1: PCA was applied ===
    if pca is not None and original_features is not None:
        print("\nRecovering original feature importances from PCA...")

        # pca.components_: shape (n_components, n_original_features)
        # model.coef_: shape (1, n_components)
        original_importances = np.dot(pca.components_.T, coefs)

        coef_df = pd.DataFrame({
            'Feature': original_features,
            'Importance': np.abs(original_importances)
        }).sort_values(by='Importance', ascending=False)

    # === Case 2: No PCA ===
    else:
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(coefs)
        }).sort_values(by='Importance', ascending=False)

    print("\nTop 30 Important Features:")
    print(coef_df.head(100))
    return coef_df


########################################### Model Traininig ############################################################

lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000)

#L2_regularization(lr, X_train, y_train, X_test, y_test)
# Best C = 1.0

#L1_regularization(lr, X_train, y_train, X_test, y_test)
# Best C = 0.1

# PCA + L2
"""
X_train, X_test, y_train, y_test = preprocess_data(features_file="features_lite.csv", labels_file="nataliia_labels.csv")
X_train, X_test, pca = apply_pca(X_train, X_test, n_components=0.95)
original_features = X_train.columns.tolist()
L2_regularization(lr, X_train, y_train, X_test, y_test)
"""
# UVFS + L2

X_train, X_test, y_train, y_test = preprocess_data()
X_train, X_test, selected_features, feature_scores_df = apply_uvfs(X_train, X_test, y_train, k_best=100)
original_features = selected_features
pca = None
L2_regularization(lr, X_train, y_train, X_test, y_test)


end = time.time()
print("Time elapsed:", end-start)
