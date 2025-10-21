import sklearn.pipeline as skp
import sklearn.feature_selection as fs
import sklearn.svm as svm
import sklearn.model_selection as skm

def svmModel(X_train, y_train, X_test, y_test):

    pipe = skp.Pipeline([
        ("filter", fs.SelectKBest()),
        ("SVM", svm.SVC(class_weight="balanced", probability=True))
    ])

    hyperparameters={
        "filter__k" : [500, 1000, 1250, 1500, 1750],
        "SVM__C" : [10,100,125,150,175],
        "SVM__kernel" : ["linear", "poly", "rbf", "sigmoid"]
    }
    grid = skm.GridSearchCV(
        estimator=pipe,
        param_grid=hyperparameters,
        scoring="f1",
        cv=5,
        verbose=2,
        n_jobs=-1

    )
    grid.fit(X_train, y_train)
    bestFit = grid.best_estimator_
    bestHyperparameters = grid.best_params_
    print(f"The best hyperparameters selected were:   {bestHyperparameters}")

    return bestFit