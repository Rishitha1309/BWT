# Data Preprocessing Pipeline

## Overview
The data preprocessing pipeline is responsible for cleaning, transforming, and engineering features from raw traffic data before training machine learning models. It ensures data quality, handles missing values, creates meaningful features, and formats the dataset for modeling.

---

## Steps in Data Preprocessing

### 1. **Loading Raw Data**
- The raw traffic dataset (`pacific_highway_data.csv`) is loaded using **Dask** for efficient memory management.
- The dataset is then converted into a **Pandas DataFrame** for further transformations.

```python
import pandas as pd
import dask.dataframe as dd

df = dd.read_csv("data/pacific_highway_data.csv", low_memory=False)
df = df.compute()  # Convert to Pandas
```

---

### 2. **Handling Missing Values**
- Missing values in `Avg - Delay` are filled with the **mean delay** grouped by `LaneType`, `DirectionOfTravel`, and `Hour`.
- Missing values in `Avg - Volume` are **forward-filled** and **backward-filled** based on `LaneType`.

```python
# Compute mean delay per group
delay_means = df.groupby(["LaneType", "DirectionOfTravel", "Hour"], as_index=False)["Avg - Delay"].mean()

# Merge and fill missing values
df = df.merge(delay_means, on=["LaneType", "DirectionOfTravel", "Hour"], how="left", suffixes=("", "_mean"))
df["Avg - Delay"] = df["Avg - Delay"].fillna(df["Avg - Delay_mean"])
df.drop(columns=["Avg - Delay_mean"], inplace=True)

# Fill missing Avg - Volume using forward and backward fill
df["Avg - Volume"] = df.groupby("LaneType")["Avg - Volume"].transform(lambda x: x.ffill().bfill())
```

---

### 3. **Aggregating Data to 15-Minute Intervals**
- The dataset is aggregated to 15-minute intervals using the `EffectiveStart` timestamp.
- `Avg - Delay` is averaged, while `Avg - Volume` is summed within each time window.

```python
df["EffectiveStart"] = pd.to_datetime(df["EffectiveStart"], errors='coerce')
df["EffectiveStart"] = df["EffectiveStart"].dt.floor("15min")

df = df.groupby(["EffectiveStart", "DirectionOfTravel", "LaneType"], as_index=False).agg({
    "Avg - Delay": "mean",
    "Avg - Volume": "sum"
})
```

---

### 4. **Extracting Time-Based Features**
- New features are created from the `EffectiveStart` timestamp, including:
  - `Year`, `Month`, `Day`, `DayOfWeek`, `Hour`
  - `QuarterHour`, `Week`, `Season`, `IsWeekend`, `WeekendImpact`
  
```python
df["Year"] = df["EffectiveStart"].dt.year
df["Month"] = df["EffectiveStart"].dt.month
df["Day"] = df["EffectiveStart"].dt.day
df["DayOfWeek"] = df["EffectiveStart"].dt.dayofweek
df["Hour"] = df["EffectiveStart"].dt.hour
df["QuarterHour"] = df["EffectiveStart"].dt.minute // 15
df["Week"] = df["EffectiveStart"].dt.isocalendar().week
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["WeekendImpact"] = df["IsWeekend"] * df["Hour"]
```

---

### 5. **Creating Lag Features**
- Lag features are created to capture historical trends:
  - **1, 3, 6, and 12 time periods** (equivalent to 15, 45, 90, and 180 minutes)
  - Rolling window averages for **1-hour, 2-hour, and 3-hour** historical values

```python
df.sort_values(["EffectiveStart"], inplace=True)

for lag in [1, 3, 6, 12]:
    df[f"Delay_Lag_{lag}Q"] = df["Avg - Delay"].shift(lag)
    df[f"Volume_Lag_{lag}Q"] = df["Avg - Volume"].shift(lag)

for window in [4, 8, 12]:
    df[f"Delay_Rolling_{window}Q"] = df["Avg - Delay"].rolling(window, min_periods=1).mean()
    df[f"Volume_Rolling_{window}Q"] = df["Avg - Volume"].rolling(window, min_periods=1).mean()

df.dropna(inplace=True)
```

---

### 6. **Merging Holiday Data**
- A holiday dataset (`holidays.csv`) is merged to add a binary `IsHoliday` feature.

```python
holiday_file = "data/holidays.csv"
if os.path.exists(holiday_file):
    df_holidays = pd.read_csv(holiday_file)
    df_holidays["Date"] = pd.to_datetime(df_holidays["Date"])
    df["IsHoliday"] = df["EffectiveStart"].dt.date.isin(df_holidays["Date"].dt.date).astype(int)
else:
    df["IsHoliday"] = 0
```

---

### 7. **Encoding Categorical Features**
- `DirectionOfTravel` and `LaneType` are **ordinally encoded**.

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[["DirectionOfTravel", "LaneType"]] = oe.fit_transform(df[["DirectionOfTravel", "LaneType"]])
```

---

### 8. **Train-Test Split**
- The dataset is **sorted by timestamp** before splitting into **80% training** and **20% test**.

```python
train_size = int(len(df) * 0.8)
X_train = df.iloc[:train_size].drop(columns=["Avg - Delay", "Avg - Volume"])
y_train = df.iloc[:train_size][["Avg - Delay", "Avg - Volume"]]
X_test = df.iloc[train_size:].drop(columns=["Avg - Delay", "Avg - Volume"])
y_test = df.iloc[train_size:][["Avg - Delay", "Avg - Volume"]]
```

---

### 9. **Saving Processed Data**
- The processed datasets are saved as **Parquet files** for efficient storage and retrieval.

```python
X_train.to_parquet("data/x_train_original.parquet", index=False)
X_test.to_parquet("data/x_test_original.parquet", index=False)
y_train.to_parquet("data/y_train_original.parquet", index=False)
y_test.to_parquet("data/y_test_original.parquet", index=False)
```

---

## Summary
✔ **Raw Data Loaded** → ✔ **Missing Values Filled** → ✔ **Aggregated to 15-Minutes** →
✔ **Feature Engineering Done** → ✔ **Categorical Encoding** → ✔ **Train-Test Split** →
✔ **Saved for Model Training**

🚀 **Preprocessing completed successfully!**

