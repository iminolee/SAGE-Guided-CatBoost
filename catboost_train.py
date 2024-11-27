import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from df_loader import load_df
from sage_features import BaseImputer
import sage

# Hyperparameters
test_name = "sage_filtered"
hparams = {
    "model_num": 1,  # CatBoost Regressor + Feature Selection
    "data_normalize": "normal",  # Normalization type
    "iterations": 10000,  # Iterations
    "learning_rate": 0.05,  # Learning rate
    "tree_depth": 8,  # Tree depth
    "importance_threshold": 0.01,  # SAGE importance threshold
    "num_leaves": 31,  # LightGBM-specific parameter
    "feature_fraction": 0.9,  # LightGBM-specific parameter
    "n_estimators": 1000,  # XGBoost/RandomForest-specific parameter
}

# Load data using datasets.py
print("Loading data...")
feature_names, X_train, X_val, X_test, Y_train, Y_val = load_df(
    train_path="data/train.csv", test_path="data/test.csv"
)

# Normalize data
scaler = None
if hparams["data_normalize"] == "min_max":
    scaler = MinMaxScaler()
elif hparams["data_normalize"] == "normal":
    scaler = StandardScaler()

if scaler:
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

# Train initial CatBoost model
# print("Training the initial CatBoost Regressor...")
# initial_model = CatBoostRegressor(
#     iterations=hparams["iterations"],
#     depth=hparams["tree_depth"],
#     learning_rate=hparams["learning_rate"],
#     loss_function="RMSE",
#     verbose=200,
# )
# initial_model.fit(X_train, Y_train, eval_set=(X_val, Y_val), use_best_model=True)

# Model training and prediction
if hparams["model_num"] == 0:  # Linear Regression
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)

elif hparams["model_num"] == 1:  # CatBoost Regressor
    print("Training CatBoost Regressor...")
    model = CatBoostRegressor(
        iterations=hparams["iterations"],
        depth=hparams["tree_depth"],
        learning_rate=hparams["learning_rate"],
        loss_function="RMSE",
        verbose=200,
    )
    model.fit(X_train, Y_train, eval_set=(X_val, Y_val), use_best_model=True)
    Y_pred = model.predict(X_val)

elif hparams["model_num"] == 2:  # LightGBM
    print("Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=Y_train)
    val_data = lgb.Dataset(X_val, label=Y_val)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": hparams["num_leaves"],
        "learning_rate": hparams["learning_rate"],
        "feature_fraction": hparams["feature_fraction"],
    }
    model = lgb.train(
        params, train_data, valid_sets=[train_data, val_data], num_boost_round=1000, early_stopping_rounds=100
    )
    Y_pred = model.predict(X_val)

elif hparams["model_num"] == 3:  # XGBoost
    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=hparams["n_estimators"],
        max_depth=hparams["tree_depth"],
        learning_rate=hparams["learning_rate"],
        objective="reg:squarederror",
        verbosity=1,
    )
    model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], early_stopping_rounds=100, verbose=100)
    Y_pred = model.predict(X_val)

elif hparams["model_num"] == 4:  # Random Forest
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=hparams["n_estimators"])
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)

# Apply SAGE to calculate Feature Importance
print("Calculating Feature Importance using SAGE...")
num_features = X_train.shape[1]
imputer = BaseImputer(model, num_features)
estimator = sage.PermutationEstimator(imputer, loss="mse")
explanation = estimator(X_val, Y_val, batch_size=100)

# Extract Feature Importance and Filter Features
feature_importances = explanation.values
selected_features = [
    idx for idx, imp in enumerate(feature_importances) if imp >= hparams["importance_threshold"]
]
selected_feature_names = [feature_names[idx] for idx in selected_features]

# Filter data for selected features
X_train_filtered = X_train[:, selected_features]
X_val_filtered = X_val[:, selected_features]
X_test_filtered = X_test[:, selected_features]

# Retrain Model with Selected Features
print("Retraining model with selected features...")
model.fit(X_train_filtered, Y_train, eval_set=(X_val_filtered, Y_val), use_best_model=True)
Y_pred_filtered = model.predict(X_val_filtered)

# Evaluate model
mse = mean_squared_error(Y_val, Y_pred)
print(f"Validation MSE: {mse}")

# Predict using the trained model
predictions = model.predict(X_test_filtered)

# --------------------------------------------------------------
# Results and Logs Handler
# --------------------------------------------------------------
result_dir = "result"
logs_dir = "logs"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Save logs
print("Saving logs and results...")
sorted_indices = np.argsort(-feature_importances)
log_file = os.path.join(logs_dir, "training_info.txt")
with open(log_file, "w") as log:
    log.write("Feature Importances (All):\n")
    for idx in sorted_indices:
        log.write(f"{feature_names[idx]}: {feature_importances[idx]:.4f}\n")
    log.write(f"\nSelected Features (importance >= {hparams['importance_threshold']}):\n")
    log.writelines(f"{feature}\n" for feature in selected_feature_names)
    log.write(f"\nValidation MSE: {mse}\n")
    for key, value in hparams.items():
        log.write(f"{key}: {value}\n")

# Save the updated submission file
submission_df = pd.read_csv("data/submission.csv", encoding="cp949")
submission_df['Temperature'] = predictions
submission_path = os.path.join(result_dir, f"{test_name}_submission.csv")
submission_df.to_csv(submission_path, index=False, encoding="cp949")
print(f"Model trained and predictions saved to '{submission_path}'.")