from PerformanceEvaluation import evaluate_model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

########################################### FEATURE SELECTION #########################################################

# Recursive Feature Elimination
def RecursiveFS(rf, X_train, y_train, X_test, y_test):
    from sklearn.feature_selection import RFE

    # Select top N features
    rfe = RFE(estimator=rf, n_features_to_select=20, step=5)
    rfe.fit(X_train, y_train)

    # Get selected features
    selected_features = X_train.columns[rfe.support_].tolist()
    print(f"Selected features: {selected_features}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print("Recursive Feature Elimination: ")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)


# Recursive Feature Elimination with Cross Validation
def RecursiveFS_CV(rf, X_train, y_train, X_test, y_test):
    from sklearn.feature_selection import RFECV
    import matplotlib.pyplot as plt

    rfecv = RFECV(
        estimator=rf,
        step=5,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")

    # Get selected features
    selected_features = X_train.columns[rfecv.support_].tolist()
    print(f"Selected features: {selected_features}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print("Recursive Feature Elimination with Cross Validation: ")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)

    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'])
    plt.xlabel('Number of Features')
    plt.ylabel('CV Score (Recall)')
    plt.title('RFECV Feature Selection')
    plt.show()


# Univariate Feature Selection
def UnivariateFS(rf, X_train, y_train, X_test, y_test):
    from sklearn.feature_selection import SelectKBest, f_classif
    # Select top K features
    k = 20
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    # Get selected features
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features}")

    # Get scores
    scores_df = pd.DataFrame({
        'feature': X_train.columns,
        'f_score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('f_score', ascending=False)

    print(f"Top 10 features by F-score:")
    print(scores_df.head(10))

    # Transform
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print("Univariate Feature Selection: ")
    evaluate_model(rf, X_train_selected, y_train, X_test_selected, y_test)

#L1 REGULARIZATION
def L1_regularization(X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [0.01, 0.1, 1.0],  # Regularization strength
        "penalty": ["l1"],  # L1 regularization
        "solver": ["saga"]  # Solver for logistic regression
    }
    LR_L1 = GridSearchCV(
        LogisticRegression(random_state=10, class_weight='balanced', max_iter=10000),
        param_grid,
        cv=5 ,
        scoring ='f1_weighted'
    )

    # Fit GridSearchCV to training data
    LR_L1.fit(X_train, y_train)

    # Best L1 model after hyperparameter tuning
    best_L1_model = LR_L1.best_estimator_

    n_used_features = np.sum(best_L1_model.coef_ != 0)
    print(f"Number of features used (L1): {n_used_features}")

    print("Best parameters (L1):\n", LR_L1.best_params_)

    evaluate_model(best_L1_model, X_train, y_train, X_test, y_test)

# L2 REGULARIZATION
def L2_regularization(X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [0.01, 0.1, 1.0],  # Regularization strength
        "penalty": ["l2"],  # L2 regularization
        "solver": ["lbfgs"]  # Solver for logistic regression
    }
    LR_L2 = GridSearchCV(
        LogisticRegression(random_state=10, class_weight='balanced', max_iter=10000),
        param_grid,
        cv=5,
        scoring='f1_weighted'
    )

    LR_L2.fit(X_train, y_train)

    best_L2_model = LR_L2.best_estimator_
    n_used_features_L2 = np.sum(best_L2_model.coef_ != 0)
    print(f"Number of features used (L2): {n_used_features_L2}")

    print("Best parameters (L2):\n", LR_L2.best_params_)

    evaluate_model(best_L2_model, X_train, y_train, X_test, y_test)