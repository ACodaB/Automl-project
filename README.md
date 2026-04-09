# 🚀 AutoML & Experiment Tracking Platform

An end-to-end **AutoML system built with Python and Streamlit** that automates the complete machine learning workflow — from data preprocessing to deployment.

---

##  Features

###  Automated ML Pipeline
- Automatic detection of **problem type** (classification/regression)
- Intelligent **model recommendation** based on dataset size
- Modular pipeline:
  - Preprocessing (imputation, encoding, scaling)
  - Feature selection
  - Model training

---

###  Hyperparameter Optimization
- Integrated **Optuna** for efficient hyperparameter tuning
- Dataset-aware sampling strategy:
  - Tune on subset → Train on full data

---

###  Experiment Tracking
- Logs:
  - Model parameters
  - Evaluation metrics
  - Training time
  - Run ID
- Displays results in a **leaderboard UI**

---

###  Model Registry
- Versioned model saving (`v1`, `v2`, ...)
- Stores:
  - Model pipeline
  - Metadata (params, metrics, features, dataset_id)
- Supports multiple models & versions

---

### Deployment Simulation
- Load latest **dataset-specific model**
- Automatic **feature alignment**
- Predict on new unseen data
- Handles:
  - Missing columns
  - Extra columns
  - Schema mismatch

---

## Key Design Concepts

- **Tune on sample → Train on full → Evaluate on test**
- **Dataset-aware model selection**
- **Pipeline-based architecture (sklearn)**
- **Modular system design**
- **Separation of training and deployment logic**

---

## ⚙️ How It Works

1. Upload dataset (CSV)
2. Select target column
3. System detects problem type
4. Data is preprocessed automatically
5. Models are trained and tuned using Optuna
6. Results are displayed in a leaderboard
7. Best models are saved with versioning
8. Upload new data → Generate predictions

---

## Project Structure
automl_project/
│
├── app.py
├── src/
│ ├── data.py
│ ├── preprocessing.py
│ ├── feature_selection.py
│ ├── models.py
│ ├── tuning.py
│ ├── training.py
│ ├── evaluation.py
│ ├── tracking.py
│ ├── registry.py
│ └── utils.py


## How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Run the app
streamlit run app.py


## Tech Stack
- Python  
- Streamlit  
- scikit-learn  
- Optuna  
- Pandas & NumPy 

## Author
Arsh Bajaj