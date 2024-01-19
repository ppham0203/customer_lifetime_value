# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import os
import warnings
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#setup metadata
XP_TRACKING_FOLDER_ID = "CacxT6xM"
MLFLOW_EXPERIMENT_NAME = "clv-mlflow-exp"
MLFLOW_CODE_ENV_NAME = "ml_flow_py36"
SAVED_MODEL_NAME = "clv-classifier-mlflow"
EVALUATION_DATASET = "customer_data_test"
MODEL_NAME = "catboost"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#read inputs
train = dataiku.Dataset("customer_data_train")
train_df = train.get_dataframe(infer_with_pandas=False)
train_df = train_df.drop('ip_geopoint', axis=1)
#train_df['High Revenue'] = train_df['High Revenue'].astype('string')

test = dataiku.Dataset("customer_data_test")
test_df = test.get_dataframe(infer_with_pandas=False)
test_df = test_df.drop('ip_geopoint', axis=1)
#test_df['High Revenue'] = test_df['High Revenue'].astype('string')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train_df.dtypes

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Choose features to include in the model
columns_to_inlcude = ['pages_visited', 'campaign', 'Country', 'GDP_per_cap', 'age', 'price_first_item_purchased', 'gender', 'High Revenue']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
columns_to_ignore = [col for col in train_df.columns if col not in columns_to_inlcude]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# This section will sync the MLFlow experiments with Dataiku
mlflow_model_cc_transaction_fraud_folder = dataiku.Folder(XP_TRACKING_FOLDER_ID)
client = dataiku.api_client()
project = client.get_default_project()

mlflow_extension = project.get_mlflow_extension()
mlflow_handle = project.setup_mlflow(managed_folder=mlflow_model_cc_transaction_fraud_folder)

mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
mlflow_experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Name the run with current timestamp
def now_str() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M")

run_name = f"{MODEL_NAME}_{now_str()}"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Catboost likes feature-type (categorical, numeric) indices
nonint_features_indices = np.where((train_df.dtypes != np.int))[0]
nonfloat_features_indices = np.where((train_df.dtypes != np.float))[0]
categorical_features_indices = [value for value in nonint_features_indices if value in nonfloat_features_indices]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tag("model", "catboost")
        mlflow.set_tag("stage", "experimenting")
        mlflow.set_tag("run_name", run_name)
        
        X = train_df.drop('High Revenue', axis=1)
        y = train_df['High Revenue']

        nonint_features_indices = np.where((X.dtypes != np.int))[0]
        nonfloat_features_indices = np.where((X.dtypes != np.float))[0]
        categorical_features_indices = [value for value in nonint_features_indices if value in nonfloat_features_indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_data = Pool(data=X_train, label=y_train, cat_features=categorical_features_indices)
        test_data = Pool(data=X_test, label=y_test, cat_features=categorical_features_indices)

        # Hyperparameters space here
        param = {'objective'         : "Logloss",
                 "ignored_features": columns_to_ignore,
                 'learning_rate' : 0.04
                 }

        # Use MLFlow to log chosen parameters
        mlflow.log_params(param)
       

        cat_cls = CatBoostClassifier(**param)
        cat_cls.fit(train_data, eval_set = test_data, verbose=0)
        mlflow.catboost.log_model(cat_cls, artifact_path=f"{run_name}")
        preds = cat_cls.predict(X_test)
        #convert_if = lambda t: 0 if t =="low_revenue" else 1
        #preds_converted = np.array([convert_if(predsi) for predsi in preds])
        pred_labels = preds.astype('bool')
        
        roc_auc = round(roc_auc_score(y_test, pred_labels),4)
        accuracy = round(accuracy_score(y_test, pred_labels),4)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the MLFlow details of the final, best trained model
experiment_id = mlflow_experiment.experiment_id
experiment_results_df = mlflow.search_runs(experiment_id)

latest_run_results_df = experiment_results_df[experiment_results_df['tags.run_name'] == run_name]
best_run_id = latest_run_results_df.iloc[0]['run_id']
model_path = f"clv_mlflow_exp/{best_run_id}/artifacts/{run_name}"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_path

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create a new Dataiku Saved Model (if doesn't exist already)
sm_id = None
for sm in project.list_saved_models():
    if sm["name"] != SAVED_MODEL_NAME:
        continue
    else:
        sm_id = sm["id"]
        print(f"Found Saved Model {sm['name']} with id {sm['id']}")
        break

if sm_id:
    sm = project.get_saved_model(sm_id)
else:
    sm = project.create_mlflow_pyfunc_model(name=SAVED_MODEL_NAME,
                                            prediction_type=DSSPredictionMLTaskSettings.PredictionTypes.BINARY)
    sm_id = sm.id
    print(f"Saved Model not found, created new one with id {sm_id}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Import the final trained model into the Dataiku Saved Model (Green Diamond)
mlflow_version = sm.import_mlflow_version_from_managed_folder(version_id=run_name,
                                                              managed_folder=XP_TRACKING_FOLDER_ID,
                                                              path=model_path,
                                                              code_env_name=MLFLOW_CODE_ENV_NAME)

# Make this Saved Model version the active one
sm.set_active_version(mlflow_version.version_id)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Set model metadata (target name, classes,...)
mlflow_version.set_core_metadata('High Revenue', ['false', 'true'] , get_features_from_dataset=EVALUATION_DATASET)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
cacxt6xm = dataiku.Folder("CacxT6xM")
cacxt6xm_info = cacxt6xm.get_info()