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

def run_experiments(X_train, X_test, y_train, y_test,
                    selected_models, problem_type, feature_method,dataset_id):

    
    X_tune, y_tune = get_tuning_data(X_train, y_train)

    
    num_cols, cat_cols = detect_columns(X_train)
    preprocessor = build_preprocessor(X_train, num_cols, cat_cols)

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


        # evaluate
        preds = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, preds, problem_type)
        training_time = time.time() - start_time

        safe_metadata = make_json_safe({
        "model": model_name,
        "params": best_params,
        "metrics": metrics,
        "features":list(X_train.columns),
        "dataset_id":dataset_id
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