import pandas as pd
import numpy as np
import os
import time
import joblib
import fsspec
import gcsfs
from google.cloud import storage
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 🚀 **GCS Paths**
BUCKET_NAME = "bwt-project-data"
DATA_DIR = f"gs://{BUCKET_NAME}/processed_data/"
BASE_MODELS_DIR = f"gs://{BUCKET_NAME}/base_models/"
STACKED_MODELS_DIR = f"gs://{BUCKET_NAME}/stacked_models/"

# 📥 **Load Training Data**
print("📥 Loading training data from GCS...")
X_train_path = os.path.join(DATA_DIR, "x_train_original.parquet")
y_train_path = os.path.join(DATA_DIR, "y_train_original.parquet")

X_train = pd.read_parquet(X_train_path, storage_options={"anon": False})
y_train = pd.read_parquet(y_train_path, storage_options={"anon": False})
print(f"✅ Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features.")

# 📥 **Load Feature Names**
fs = gcsfs.GCSFileSystem()
FEATURE_NAMES_PATH = f"{BASE_MODELS_DIR}feature_names.pkl"
with fs.open(FEATURE_NAMES_PATH, 'rb') as features_file:
    feature_names = joblib.load(features_file)
X_train = X_train.reindex(columns=feature_names, fill_value=0)

# 📊 **Apply MinMax Scaling (Same as Base Models)**
GCS_SCALER_PATH = f"gs://{BUCKET_NAME}/{BASE_MODELS_DIR}scaler.pkl"
with fs.open(GCS_SCALER_PATH, 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)
X_train = pd.DataFrame(scaler.transform(X_train), columns=feature_names)

# 📊 **Generate Base Model Predictions**
print("\n🚀 Generating base model predictions for stacking...")

base_models = {
    "XGBoost": ["xgboost_avg_-_delay.pkl", "xgboost_avg_-_volume.pkl"],
    "LightGBM": ["lightgbm_avg_-_delay.pkl", "lightgbm_avg_-_volume.pkl"],
    "CatBoost": ["catboost_avg_-_delay.pkl", "catboost_avg_-_volume.pkl"]
}

X_train_stacked = pd.DataFrame()

for model_name, file_paths in base_models.items():
    print(f"🚀 Generating predictions from {model_name}...")

    # Load models from GCS
    MODEL_DELAY_PATH = f"{BASE_MODELS_DIR}{file_paths[0]}"  # Construct delay model path
    MODEL_VOLUME_PATH = f"{BASE_MODELS_DIR}{file_paths[1]}"  # Construct volume model path

    print(f"DELAY_MODEL_PATH: {MODEL_DELAY_PATH}")
    print(f"VOLUME_MODEL_PATH: {MODEL_VOLUME_PATH}")
    with fs.open(MODEL_DELAY_PATH, "rb") as delay_model:
        model_delay = joblib.load(delay_model)
    with fs.open(MODEL_VOLUME_PATH, "rb") as volume_model:
        model_volume = joblib.load(volume_model)
    # Make predictions
    X_train_stacked[f"{model_name}_Delay"] = model_delay.predict(X_train)
    X_train_stacked[f"{model_name}_Volume"] = model_volume.predict(X_train)

print("✅ Base model predictions generated for stacking!")

# 🚀 **Train Ridge Regression for Stacking**
stacked_models = {}
results = {}

for target in ["Avg - Delay", "Avg - Volume"]:
    print(f"\n🚀 Training Ridge Regression model for {target}...")

    # Train Ridge Regression Model
    ridge_model = Ridge(alpha=1.0)  # Regularization parameter (can tune later)
    ridge_model.fit(X_train_stacked, y_train[target])

    # Save model to GCS
    STACKED_MODEL_PATH = f"{STACKED_MODELS_DIR}ridge_stacked_{target.replace(' ', '_').lower()}.pkl"  # Construct stacked model path
    with fs.open(STACKED_MODEL_PATH, "wb") as stacked_model:
        joblib.dump(ridge_model, stacked_model)
    
    stacked_models[target] = ridge_model

    # Evaluate model performance on training data
    y_pred_train = ridge_model.predict(X_train_stacked)
    rmse_train = np.sqrt(mean_squared_error(y_train[target], y_pred_train))
    r2_train = r2_score(y_train[target], y_pred_train) * 100  # Convert to percentage

    results[target] = {
        "RMSE (Train)": rmse_train,
        "Accuracy (Train)": r2_train
    }

    print(f"✅ Ridge Stacked Model for {target} trained: RMSE = {rmse_train:.2f}, Accuracy = {r2_train:.2f}%")
    print(f"✅ Model saved at {STACKED_MODEL_PATH}")

# 📊 **Print Training Performance**
print("\n✅ Stacked Ridge Training Performance:")
for target, metrics in results.items():
    print(f"{target}: RMSE (Train) = {metrics['RMSE (Train)']:.2f}, Accuracy (Train) = {metrics['Accuracy (Train)']:.2f}%")

print("\n🚀 Ridge Regression Stacking Training Completed!")
