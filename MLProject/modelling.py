import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# HAPUS tracking URI localhost
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# FIX PATH (INI KUNCI)
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "namadataset_preprocessing", "data_clean.csv")

df = pd.read_csv(file_path)

# split data
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# mulai run MLflow
with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # log metric
    mlflow.log_metric("accuracy", acc)

    # log model
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)