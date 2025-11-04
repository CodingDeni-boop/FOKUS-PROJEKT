import joblib
import json
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from DataPreprocessing import preprocess_data
from sklearn.linear_model import LinearRegression

def save_grid_as_pkl(name : str, grid : GridSearchCV, columns : list, random_state : int, folder ="SVM_(hyper)parameters"):

    print("dumping hyperparameters")
    with open("./"+folder+"/"+name+"_hyperparameters.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)

    print("dumping model")
    joblib.dump(grid.best_estimator_,"./"+folder+"/"+name+"_model.pkl")

    print("dumping dataset columns")
    joblib.dump(columns,"./"+folder+"/"+name+"_columns.pkl")

def save_model_as_pkl(name : str, model : LinearRegression, columns : list, random_state : int, folder ="SVM_(hyper)parameters"):

    print("dumping model")
    joblib.dump(model,"./"+folder+"/"+name+"_model.pkl")

    print("dumping dataset columns")
    joblib.dump(columns,"./"+folder+"/"+name+"_columns.pkl")

def load_model_from_pkl(name : str, random_state : int,features_path = "processed_features.csv", labels_path= "processed_labels.csv", folder ="SVM_(hyper)parameters"):
    print("loading model...")
    model : SVC = joblib.load("./SVM_(hyper)parameters/"+name+"_model.pkl")

    print("loading dataset...")
    columns = joblib.load("./SVM_(hyper)parameters/"+name+"_columns.pkl")

    X_train, X_test, y_train, y_test = preprocess_data(
        features_file=features_path,
        labels_file=labels_path,
        random_state=random_state
    )

    X_train = X_train[columns]
    X_test = X_test[columns]
    
    return X_train, X_test, y_train, y_test, model

