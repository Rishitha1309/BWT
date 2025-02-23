import pandas as pd
import numpy as np
import os
import time
import gcsfs
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 🚀 **GCS Paths**
BUCKET_NAME = "bwt-project-data"
PROCESSED_DATA_PATH = f"gs://{BUCKET_NAME}/processed_data/"
MODEL_STORAGE_PATH = f"gs://{BUCKET_NAME}/base_models/"

# 📥 **Load Data from GCS**
print("📥 Loading data from GCS...")
load_start = time.time()
X_train = pd.read_parquet(f"{PROCESSED_DATA_PATH}x_train_original.parquet", storage_options={"anon": False})
y_train = pd.read_parquet(f"{PROCESSED_DATA_PATH}y_train_original.parquet", storage_options={"anon": False})
X_test = pd.read_parquet(f"{PROCESSED_DATA_PATH}x_test_original.parquet", storage_options={"anon": False})
y_test = pd.read_parquet(f"{PROCESSED_DATA_PATH}y_test_original.parquet", storage_options={"anon": False})
print(f"✅ Data loaded in {time.time() - load_start:.2f} seconds.")

# 🚀 **Ensure Test Set Has Same Columns as Train**
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 🚀 **Apply MinMax Scaling**
print("📊 Applying MinMax scaling...")
scaling_start = time.time()
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# Save scaler to GCS
GCS_SCALER_PATH = f"gs://{BUCKET_NAME}/{MODEL_STORAGE_PATH}scaler.pkl"
fs = gcsfs.GCSFileSystem()
with fs.open(GCS_SCALER_PATH, 'wb') as scaler_file:
    joblib.dump(scaler, scaler_file)
print(f"✅ Scaling completed and saved in {GCS_SCALER_PATH} in {time.time() - scaling_start:.2f} seconds.")

# 🚀 **Save Final Feature Names**
feature_names = X_train_scaled.columns.tolist()
FEATURE_NAMES_PATH = f"{MODEL_STORAGE_PATH}feature_names.pkl"
with fs.open(FEATURE_NAMES_PATH, 'wb') as features_file:
    joblib.dump(feature_names, features_file)
print(f"✅ Feature names saved to {FEATURE_NAMES_PATH} ({len(feature_names)} features)")

# 🚀 **Define Models**
models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, subsample=0.7, colsample_bytree=0.8, eval_metric="rmse", random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, loss_function="RMSE", random_seed=42, verbose=100)
}

# 🚀 **Train & Evaluate Models**
print("🚀 Starting model training...")
results = {}
for name, model in models.items():
    for target in ["Avg - Delay", "Avg - Volume"]:
        model_filename = f"{name.replace(' ', '_').lower()}_{target.replace(' ', '_').lower()}.pkl"
        model_path = f"{MODEL_STORAGE_PATH}{model_filename}"

        print(f"🚀 Training {name} for {target}...")
        start_time = time.time()

        model.fit(X_train_scaled, y_train[target])  # ✅ Train on scaled data
        y_pred = model.predict(X_test_scaled)  # ✅ Predict on scaled test data

        # 🚀 **Calculate RMSE**
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
        results[f"{name} ({target})"] = {"Test RMSE": rmse, "Training Time (s)": time.time() - start_time}

        # 💾 **Save Model to GCS**
        with fs.open(model_path, 'wb') as model_file:
            joblib.dump(model, model_file)
        print(f"✅ {name} ({target}) model saved to {model_path}")

# 🚀 **Print Model Performance**
print("\n✅ Model Performance:")
for model, metrics in results.items():
    print(f"{model}: Test RMSE = {metrics['Test RMSE']:.2f}, Training Time = {metrics['Training Time (s)']:.2f}s")

print("🚀 Model training completed successfully!")
