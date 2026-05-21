import time
import uuid
import streamlit as st
from src.evaluation import evaluate_model
from src.models import get_models
from src.tuning import run_optuna
from src.tracking import log_experiment
from src.data import get_tuning_data
from src.util import detect_columns
from src.preprocessing import build_preprocessor, build_pipeline
from src.registry import save_model,make_json_safe
import numpy as np
import joblib
import os

def run_experiments(X_train, X_test, y_train, y_test,
                    selected_models, problem_type, feature_method,dataset_id):

    
    num_cols, cat_cols = detect_columns(X_train)
    (preprocessor,X_train,cols_to_drop,indicator_cols) = (build_preprocessor(X_train, num_cols, cat_cols))

    X_test=X_test.drop(columns=cols_to_drop,errors="ignore")

    for col in indicator_cols:
        X_test[f"{col}_missing"]=(X_test[col].isnull().astype(int))
        
    X_tune, y_tune = get_tuning_data(X_train, y_train)

    X_tune=X_tune.drop(columns=cols_to_drop,errors="ignore")

    for col in indicator_cols:
        X_tune[f"{col}_missing"]=(X_tune[col].isnull().astype(int))

    models_dict = get_models(problem_type)
    results = []

    progress = st.progress(0)

    for i, model_name in enumerate(selected_models):

        start_time = time.time()
        run_id = str(uuid.uuid4())

        model = models_dict[model_name]

        pipeline = build_pipeline(
            preprocessor,
            model,
            feature_method,
            problem_type
        )

        #dataset-based trials
        n_samples = X_train.shape[0]
        n_trials = 15 if n_samples < 10000 else 10 if n_samples < 100000 else 5

        #FIXED Optuna call
        study = run_optuna(
            pipeline,
            model_name,
            X_tune,
            y_tune,
            problem_type,
            n_trials
        )

        best_params = study.best_params

        # train final model
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)

        # Create artifacts directory if missing
        os.makedirs("artifacts", exist_ok=True)
        pipeline_path = (f"artifacts/{model_name}_pipeline.pkl")

        joblib.dump(pipeline,pipeline_path)



        # evaluate
        preds = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, preds, problem_type)
        training_time = time.time() - start_time

        # store small sample of training data (for drift detection)
        sample_size = int(0.05 * len(X_train))

        # keep within limits(upper and lower bounds)
        sample_size = max(100, sample_size)   # minimum datasize
        sample_size = min(500, sample_size)   # maximum datasize

        train_sample = X_train.sample(
            sample_size,
            random_state=42
        ).to_dict(orient="list")

        safe_metadata = make_json_safe({
        "model": model_name,
        "params": best_params,
        "metrics": metrics,
        "features":list(X_train.columns),
        "cols_to_drop" : cols_to_drop,
        "indicator_cols" : indicator_cols,
        "dataset_id":dataset_id,
        "train_sample": train_sample,
        "pipeline_path" : pipeline_path
        
        })
        
        #save model and metadata
        model_path, version = save_model(
            pipeline,
            model_name,
            safe_metadata
        )


        run_data = {
            "run_id": run_id,
            "model": model_name,
            "problem_type": problem_type,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "feature_method": feature_method,
            "params": best_params,
            "metrics": metrics,
            "model_path": model_path,
            "version": version,
            "training_time": round(training_time, 2)
        }

        log_experiment(run_data)

        results.append(run_data)

        progress.progress((i + 1) / len(selected_models)) 

    return results 