import pandas as pd
import numpy as np
import os
import time
import joblib
import gcsfs
import dask.dataframe as dd
from google.cloud import storage
from sklearn.preprocessing import OrdinalEncoder

# 🚀 **File Paths in GCS**
BUCKET_NAME = "bwt-project-data"
RAW_DATA_PATH = f"gs://{BUCKET_NAME}/raw_data/pacific_highway_data.csv"
HOLIDAY_FILE_PATH = f"gs://{BUCKET_NAME}/raw_data/holidays.csv"
PROCESSED_FOLDER = f"gs://{BUCKET_NAME}/processed_data"

start_time = time.time()
print("🚀 Starting preprocessing...")

### **1️⃣ Load Data from GCS**
load_start = time.time()
df = dd.read_csv(RAW_DATA_PATH, storage_options={"anon": False}, low_memory=False)
df = df.compute()  # Convert Dask to Pandas
print(f"✅ Loaded raw traffic data in {time.time() - load_start:.2f} seconds.")

### **2️⃣ Convert EffectiveStart to datetime**
df["EffectiveStart"] = pd.to_datetime(df["EffectiveStart"], errors="coerce")
print("✅ Converted EffectiveStart to datetime.")

### **3️⃣ Drop 'Crossing' Column if it Exists**
if "Crossing" in df.columns and df["Crossing"].nunique() == 1:
    df.drop(columns=["Crossing"], inplace=True)
print("✅ Dropped 'Crossing' column.")

### **4️⃣ Handle Missing Values**
df["Hour"] = df["EffectiveStart"].dt.hour

# Compute mean delay per group for missing values
delay_means = df.groupby(["LaneType", "DirectionOfTravel", "Hour"], as_index=False)["Avg - Delay"].mean()

# Merge & Fill missing Avg - Delay
df = df.merge(delay_means, on=["LaneType", "DirectionOfTravel", "Hour"], how="left", suffixes=("", "_mean"))
df["Avg - Delay"].fillna(df["Avg - Delay_mean"], inplace=True)
df.drop(columns=["Avg - Delay_mean"], inplace=True)

# Fill missing Avg - Volume using forward-fill and backward-fill
df["Avg - Volume"] = df.groupby("LaneType")["Avg - Volume"].transform(lambda x: x.ffill().bfill())

# Handle different data types separately
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(0, inplace=True)  # Fill missing numeric values with 0
print("✅ Filled missing values.")

### **5️⃣ Aggregate Data to 15-Minute Intervals**
df["EffectiveStart"] = df["EffectiveStart"].dt.floor("15min")
agg_funcs = {"Avg - Delay": "mean", "Avg - Volume": "sum"}
df = df.groupby(["EffectiveStart", "DirectionOfTravel", "LaneType"], as_index=False).agg(agg_funcs)
print("✅ Aggregated data to 15-minute intervals.")

### **6️⃣ Extract Time Features**
df["Year"] = df["EffectiveStart"].dt.year
df["Month"] = df["EffectiveStart"].dt.month
df["Day"] = df["EffectiveStart"].dt.day
df["DayOfWeek"] = df["EffectiveStart"].dt.dayofweek
df["Hour"] = df["EffectiveStart"].dt.hour

df["Minute"] = df["EffectiveStart"].dt.minute
df["QuarterHour"] = df["Minute"] // 15
df["Week"] = df["EffectiveStart"].dt.isocalendar().week
df["Season"] = df["EffectiveStart"].dt.month.map(lambda x: 1 if x in [12, 1, 2] 
                                                  else 2 if x in [3, 4, 5] 
                                                  else 3 if x in [6, 7, 8] 
                                                  else 4)

# 📅 **Time of Day Buckets**
df["TimeOfDay"] = np.select(
    [df["Hour"] < 12, (df["Hour"] >= 12) & (df["Hour"] < 16), df["Hour"] >= 16],
    [0, 1, 2],  
    default=0
)
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["WeekendImpact"] = df["IsWeekend"] * df["Hour"]
print("✅ Extracted time-based features.")

### **7️⃣ Create Lag Features**
df.sort_values(["EffectiveStart"], inplace=True)

# Lag values based on 15-minute aggregation
for lag in [1, 3, 6, 12]:  # Lag by 15 min, 45 min, 90 min, 180 min
    df[f"Delay_Lag_{lag}Q"] = df["Avg - Delay"].shift(lag)
    df[f"Volume_Lag_{lag}Q"] = df["Avg - Volume"].shift(lag)

# Rolling window features
for window in [4, 8, 12]:  # Rolling averages for past 1h, 2h, 3h
    df[f"Delay_Rolling_{window}Q"] = df["Avg - Delay"].rolling(window, min_periods=1).mean()
    df[f"Volume_Rolling_{window}Q"] = df["Avg - Volume"].rolling(window, min_periods=1).mean()

df.dropna(inplace=True)  # Drop NaN rows caused by lags
print("✅ Created lag & rolling window features.")

### **8️⃣ Merge Holiday Data**
if os.path.exists(HOLIDAY_FILE_PATH):
    df_holidays = pd.read_csv(HOLIDAY_FILE_PATH)
    df_holidays["Date"] = pd.to_datetime(df_holidays["Date"])
    df["IsHoliday"] = df["EffectiveStart"].dt.date.isin(df_holidays["Date"].dt.date).astype(int)
else:
    df["IsHoliday"] = 0
print("✅ Merged holiday data.")

### **9️⃣ Encode Categorical Features**
categorical_features = ["DirectionOfTravel", "LaneType"]
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[categorical_features] = oe.fit_transform(df[categorical_features])

# Save encoder to GCS
GCS_ENCODER_PATH = f"gs://{BUCKET_NAME}/processed_data/ordinal_encoder.pkl"
fs = gcsfs.GCSFileSystem()
with fs.open(GCS_ENCODER_PATH, 'wb') as f:
    joblib.dump(oe, f)
print(f"✅ Saved ordinal encoder to {GCS_ENCODER_PATH}")

### **🔟 Train-Test Split**
df.drop(columns=["EffectiveStart"], inplace=True)  # Remove timestamp before splitting
train_size = int(len(df) * 0.8)  # 80% Train, 20% Test
X_train, X_test = df.iloc[:train_size].drop(columns=["Avg - Delay", "Avg - Volume"]), df.iloc[train_size:].drop(columns=["Avg - Delay", "Avg - Volume"])
y_train, y_test = df.iloc[:train_size][["Avg - Delay", "Avg - Volume"]], df.iloc[train_size:][["Avg - Delay", "Avg - Volume"]]
print("✅ Split data into train and test sets.")

### **1️⃣1️⃣ Save Processed Data to GCS**
def save_to_gcs(df_input, filename):
    output_path = f"{PROCESSED_FOLDER}/{filename}"
    df_input.to_parquet(output_path, index=False, storage_options={"anon": False})
    print(f"✅ Saved {filename} to {output_path}")

save_to_gcs(X_train, "x_train_original.parquet")
save_to_gcs(X_test, "x_test_original.parquet")
save_to_gcs(y_train, "y_train_original.parquet")
save_to_gcs(y_test, "y_test_original.parquet")

### **🛑 End Timing**
execution_time = time.time() - start_time
print(f"🚀 Preprocessing completed in {execution_time:.2f} seconds.")
