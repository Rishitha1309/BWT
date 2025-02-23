import pandas as pd
import numpy as np
import os
import joblib
import fsspec
import gcsfs
from google.cloud import storage
from sklearn.metrics import mean_squared_error, r2_score

# 🚀 **GCS Paths**
BUCKET_NAME = "bwt-project-data"
DATA_DIR = f"gs://{BUCKET_NAME}/processed_data/"
BASE_MODELS_DIR = f"gs://{BUCKET_NAME}/base_models/"
STACKED_MODELS_DIR = f"gs://{BUCKET_NAME}/stacked_models/"
PREDICTIONS_DIR = f"gs://{BUCKET_NAME}/stacked_predictions/"

# 📥 **Load Test Data**
print("📥 Loading test data from GCS...")
X_test_path = os.path.join(DATA_DIR, "x_test_original.parquet")
y_test_path = os.path.join(DATA_DIR, "y_test_original.parquet")

X_test = pd.read_parquet(X_test_path, storage_options={"anon": False})
y_test = pd.read_parquet(y_test_path, storage_options={"anon": False})
print(f"✅ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features.")

# 📂 **Load Feature Names**
fs = gcsfs.GCSFileSystem()
FEATURE_NAMES_PATH = f"{BASE_MODELS_DIR}feature_names.pkl"
with fs.open(FEATURE_NAMES_PATH, 'rb') as features_file:
    feature_names = joblib.load(features_file)
X_test = X_test.reindex(columns=feature_names, fill_value=0)

# 📊 **Apply MinMax Scaling (Same as Training)**
GCS_SCALER_PATH = f"gs://{BUCKET_NAME}/{BASE_MODELS_DIR}scaler.pkl"
with fs.open(GCS_SCALER_PATH, 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# 📊 **Generate Base Model Predictions**
print("\n🚀 Generating base model predictions for stacking...")

base_models = {
    "XGBoost": ["xgboost_avg_-_delay.pkl", "xgboost_avg_-_volume.pkl"],
    "LightGBM": ["lightgbm_avg_-_delay.pkl", "lightgbm_avg_-_volume.pkl"],
    "CatBoost": ["catboost_avg_-_delay.pkl", "catboost_avg_-_volume.pkl"]
}

X_test_stacked = pd.DataFrame()

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
    X_test_stacked[f"{model_name}_Delay"] = model_delay.predict(X_test)
    X_test_stacked[f"{model_name}_Volume"] = model_volume.predict(X_test)

print("✅ Base model predictions generated for stacking!")

# 🚀 **Load Stacked Ridge Models**
stacked_models = {}
for target in ["Avg - Delay", "Avg - Volume"]:
    STACKED_MODEL_PATH = f"{STACKED_MODELS_DIR}ridge_stacked_{target.replace(' ', '_').lower()}.pkl"  # Construct stacked model path

    # Load from GCS
    with fs.open(STACKED_MODEL_PATH, "rb") as stacked_model:
        stacked_models[target] = joblib.load(stacked_model)
    print(f"✅ Loaded Ridge Regression model for {target}.")

# 📊 **Make Final Predictions Using Stacked Ridge Model**
predictions = {}

for target in ["Avg - Delay", "Avg - Volume"]:
    print(f"\n🚀 Making final predictions for {target} using Ridge Regression...")

    model = stacked_models[target]
    y_pred = model.predict(X_test_stacked)

    # Compute RMSE & Accuracy
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
    accuracy = r2_score(y_test[target], y_pred) * 100  # Convert to percentage

    predictions[target] = {
        "Predictions": y_pred,
        "RMSE": rmse,
        "Accuracy": accuracy
    }

    print(f"✅ Ridge Stacked Model Performance for {target}: RMSE = {rmse:.2f}, Accuracy = {accuracy:.2f}%")

    # Save Predictions to GCS
    predictions_df = pd.DataFrame({
        f"Actual {target}": y_test[target].values,
        f"Predicted {target}": y_pred
    })
    
    PREDICTIONS_PATH = f"{PREDICTIONS_DIR}ridge_stacked_predictions_{target.replace(' ', '_').lower()}.csv"
    with fs.open(PREDICTIONS_PATH, "w") as predictions_file:
        predictions_df.to_csv(predictions_file, index=False)

    print(f"✅ Predictions saved to {PREDICTIONS_PATH}")

print("\n🚀 Ridge Regression Stacking Predictions Completed!")
