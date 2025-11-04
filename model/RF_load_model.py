from save_and_load_as_pkl import load_model_from_pkl
import pandas as pd


X_train, X_test, y_train, y_test, model = load_model_from_pkl(name = "rf",random_state=42,
                                                              features_path = "processed_features.csv",
                                                              labels_path="processed_labels.csv",
                                                              folder = "RF_model")



y_test = pd.Series(y_test,name = "label")
y_pred = pd.Series(model.predict(X_test),name="label")
y_pred.to_csv("./rf_prediction.csv",index=0)
y_test.to_csv("./rf_test.csv",index=0)