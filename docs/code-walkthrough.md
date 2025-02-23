# Code Walk-Through: Traffic Congestion Prediction Model

**Table of Contents**
- [Model Training Flow](#model-training-flow)
    - [How to Run](#how-to-run)
    - [Final Model Evaluation](#final-model-evaluation)
- [Prediction Flow](#prediction-flow)
   - [Prediction Local Run](#prediction-local-run)

## Model Training Flow:

## How to Run

1. **Prerequisites**  
   - Python 3.8+
   - Vertex AI workbench

2. **Setup**  
   ```bash
   pip install -r requirements.txt #install project dependencies
   ```

3. **Run** 
   ```bash
   python preprocess_data.py #preprocess the raw data

4. **Run**  
   ```bash
   python train_base_models.py  # Train and evaluate models
   ```
   This will output trained models in the `base_models/` directory along with feature names.

5. **Run**  
   ```bash
   python stacked_training.py  # Train and evaluate models
   ```
   This will output trained models in the `models_stacked/` directory along with feature names.

6. **Run**  
   ```bash
   python stacked_predict.py  # Train and evaluate models
   ```
   This will output trained models in the `stacked_predictions/` directory along with feature names.

---

### Improvements Summary

| **Aspect**              | **v1**                          | **v2**                          | **v3** (Current)                |
|--------------------------|---------------------------------|---------------------------------|---------------------------------|
| **Algorithm**            | Linear Regression              | RandomForest                   | **XGBoost, LightGBM, CatBoost** |
| **Data Aggregation**     | 5-min intervals                | 15-min intervals               | **Rolling Averages Added**      |
| **Zero Handling**        | Ignored                        | Simple Replacement             | **Zero Indicator Features**     |
| **Feature Engineering**  | Basic Temporal Features        | Rolling Averages               | **Volume/Delay Ratios**         |
| **Evaluation**           | Simple Split                   | Cross-validation               | **CV + Scaling Integration**    |

---

**Key Results:**
- Improved RMSE for both delay and volume predictions
- Reduced overfitting by incorporating rolling averages and zero-indicator variables

## Prediction Flow

```mermaid
flowchart TD
    A[Input past data] --> B[Load data]
    B --> C{Preproces data}
    C -->|| D[Convert to DataFrame]
    D --> E[Preprocess Features]
    E --> F[Align Features with Training Data]
    F --> G[Load Trained Model]
    G --> H[Make Prediction]
    H --> I[Return Response]
```


---

This document outlines the full ML workflow from **training** to **prediction** on Google Cloud Platform 🚀.

