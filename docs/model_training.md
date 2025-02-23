# Model Training

## Overview
This document outlines the model training pipeline used for predicting border traffic congestion. The pipeline trains multiple machine learning models, evaluates their performance, and saves the trained models for inference.

---

## Models Used
The following machine learning models are trained for predicting **Avg - Delay** and **Avg - Volume**:

- **XGBoost** (Extreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting Machine)
- **CatBoost** (Categorical Boosting)
- **Ridge Regression Stacking** (Final stacking model)

---

## Training Pipeline

### **1. Load Preprocessed Data**
The training data is loaded from preprocessed `.parquet` files:

- `x_train_original.parquet`: Training features
- `y_train_original.parquet`: Training target values

```python
import pandas as pd

X_train = pd.read_parquet("data/x_train_original.parquet")
y_train = pd.read_parquet("data/y_train_original.parquet")
```

### **2. Ensure Feature Consistency**
We ensure that test data has the same feature set as training data:

```python
import joblib

feature_names = joblib.load("models/feature_names.pkl")
X_train = X_train.reindex(columns=feature_names, fill_value=0)
```

### **3. Apply Feature Scaling**
Since tree-based models do not require scaling, but Ridge Regression does, we apply **MinMax Scaling**:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
joblib.dump(scaler, "models/scaler.pkl")
```

### **4. Train Base Models**
Each model is trained separately for **Avg - Delay** and **Avg - Volume** predictions.

#### **XGBoost Training Example**:
```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error

xgb_model_delay = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10)
xgb_model_delay.fit(X_train_scaled, y_train["Avg - Delay"])

joblib.dump(xgb_model_delay, "models/xgboost_avg_-_delay.pkl")
```

Similarly, other models like LightGBM and CatBoost are trained and saved.

### **5. Generate Base Model Predictions for Stacking**
We generate predictions from base models to create inputs for the stacking model:

```python
X_train_stacked = pd.DataFrame()
X_train_stacked["XGBoost_Delay"] = xgb_model_delay.predict(X_train_scaled)
X_train_stacked["XGBoost_Volume"] = xgb_model_volume.predict(X_train_scaled)
```

### **6. Train Ridge Regression Stacking Model**
A Ridge Regression model is trained on the base model predictions:

```python
from sklearn.linear_model import Ridge

ridge_model_delay = Ridge(alpha=1.0)
ridge_model_delay.fit(X_train_stacked, y_train["Avg - Delay"])

joblib.dump(ridge_model_delay, "stacked_models/ridge_stacked_avg_-_delay.pkl")
```

---

## Model Evaluation
### **Performance Metrics**
We use the following metrics to evaluate model performance:

- **Root Mean Squared Error (RMSE)**
- **R-Squared Score (Accuracy %)**

### **Evaluation Example**
```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred_delay = ridge_model_delay.predict(X_train_stacked)
rmse_delay = np.sqrt(mean_squared_error(y_train["Avg - Delay"], y_pred_delay))
r2_delay = r2_score(y_train["Avg - Delay"], y_pred_delay) * 100
```

---

## Training Results
| Model        | RMSE (Avg - Delay) | Accuracy (Avg - Delay) | RMSE (Avg - Volume) | Accuracy (Avg - Volume) |
|-------------|------------------|---------------------|------------------|---------------------|
| XGBoost     | **5.64**         | 92.1%              | **5.26**         | 94.3%              |
| LightGBM    | 6.21             | 90.3%              | 5.66             | 92.7%              |
| CatBoost    | 6.12             | 91.2%              | 5.51             | 93.2%              |
| Ridge Stacked | **3.38**         | **94.8%**          | **4.43**         | **95.1%**          |

---

## Conclusion
The Ridge Regression Stacked model outperforms individual base models by leveraging their combined predictions. This ensures better generalization and robustness in predicting border traffic congestion.

