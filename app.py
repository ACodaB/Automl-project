import streamlit as st
from src.data import load_data,split_data
from src.util import (detect_problem_type, results_to_df,detect_columns,
    suggest_target,data_report)
from src.training import run_experiments
from src.models import get_models, filter_models, recommend_models
from src.registry import (load_model_metadata,
                          get_registered_models,load_model_by_dataset)
import hashlib
from src.monitoring import log_prediction
from src.dashboard import show_dashboard
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import joblib

st.title("AutoML System")

filename = st.file_uploader("Upload CSV", type=["csv"])

if filename:
    result = load_data(filename)

    if isinstance(result, str):
        st.error(result)
    else:
        df = result
        

        dataset_id = hashlib.md5(
            df.to_csv(index=False).encode()
        ).hexdigest()
        st.success("File loaded successfully!")

 
        num_cols, cat_cols = detect_columns(df)

        original_n_samples = df.shape[0]
        original_n_features = df.shape[1]

        st.subheader("Column Types")
        st.write("Numerical Columns:", num_cols)
        st.write("Categorical Columns:", cat_cols)

        report = data_report(df)
        st.subheader("Dataset Summary")
        st.write("Shape:", report["Data shape"])
        st.write("Missing Values:", report["Missing values"])
        st.write("Data Types:", report["Column datatype"])

        suggested = suggest_target(df)
        target = st.selectbox("Select Target Column", df.columns,index=df.columns.get_loc(suggested))
        removed=st.multiselect("Select unwanted Columns",options=list(df.columns),
                            max_selections=len(df.columns)-2)

        X = df.drop(columns=[target]+removed)
        y = df[target]

        user_choice = st.selectbox(
            "Confirm Problem Type",
            ["auto", "classification", "regression"]
        )

        problem_type_auto = detect_problem_type(y)
        if user_choice == "auto":
            problem_type = problem_type_auto
        else:
            problem_type = user_choice
        st.info(f"Problem type is: {problem_type}")


        method = st.selectbox(
            "Feature Selection Method",
            ["none", "filter", "model", "hybrid"]
        )

        X_train, X_test, y_train, y_test = split_data(X, y)

        st.info(f"""
        Original Data: {original_n_samples} rows  
        Training Data: {len(X_train)} rows
        Test Data: {len(X_test)} rows
        """)


        def full_training_ui(y, X_train, X_test, y_train, y_test, feature_method):

            problem_type = detect_problem_type(y)
            st.info(f"Detected: {problem_type}")

            model_dict = get_models(problem_type)
            model_list = list(model_dict.keys())

            model_list = filter_models(original_n_samples, model_list)
            recommended = recommend_models(original_n_samples,original_n_features)

            selected_models = st.multiselect(
                "Select Models",
                model_list,
                default=[m for m in recommended if m in model_list]
            )
            if "training_done" not in st.session_state:
                st.session_state.training_done = False 

            if "results" not in st.session_state:
                st.session_state.results = None

            if "trained_models" not in st.session_state:
                st.session_state.trained_models = []


            if st.button("Run Experiments"):

                results = run_experiments(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    selected_models,
                    problem_type,
                    feature_method,
                    dataset_id
                )
                #to make deployement part stay after button click
                st.session_state.results = results
                trained_models=[r["model"] for r in results]
                st.session_state.trained_models = trained_models
                st.session_state.training_done = True



            if st.session_state.training_done:

                df_results = results_to_df(st.session_state.results)

                st.subheader("Results")
                st.dataframe(df_results)

                if problem_type == "classification":
                    best = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
                else:
                    best = df_results.sort_values(by="R2 Score", ascending=False).iloc[0]

                st.success(f"Best Model: {best['Model']}")

                
                deployment_ui(target,dataset_id)
                # safe_df=make_streamlit_safe(df_results)
                
                
        def deployment_ui(target,dataset_id):

            st.subheader("Make Predictions")

            #get models from registry
            registered_models = get_registered_models()

            valid_models=[]

            # 🔹 filter based on dataset_id
            for model_name in registered_models:
                model_obj,best_version=load_model_by_dataset(model_name,dataset_id)
                if model_obj is not None:
                    valid_models.append(model_name)

            #get models from session
            session_models = st.session_state.get("trained_models", [])

            #combine both (no duplicates)
            available_models = list(set(valid_models + session_models))

            with st.expander("Model Info"):
                st.write("Registered Models:", registered_models)
                st.write("Session Models:", session_models)

            if not available_models:
                st.warning("No models available for this dataset. Train a model first.")
                return
            

            model_name_input = st.selectbox(
                "Select Model",
                available_models,
                key="deploy_model"
            )

            new_file = st.file_uploader(
                "Upload New Data",
                type=["csv"],
                key="predict_file"
            )

            if new_file:
                train_df=None
                new_df = load_data(new_file)
                if isinstance(new_df, str):
                    st.error(new_df)
                    return
                if new_df.empty:
                    st.error("Uploaded file is empty")
                    return
                st.success("New data loaded")
                st.write("Preview of uploaded data:")
                st.write(new_df.head().to_dict(orient="records"))


                # remove target if exists
                if target in new_df.columns:
                    new_df = new_df.drop(columns=[target])

                if st.button("Predict", key="predict_btn"):

                    _, version = load_model_by_dataset(model_name_input,dataset_id)
                    metadata = load_model_metadata(model_name_input, version)

                    pipeline_path = metadata.get("pipeline_path")
                    pipeline = joblib.load(pipeline_path) if pipeline_path else None

                    if pipeline is None:
                        st.error("No saved pipeline found")
                        return
                    st.info(f"Using model: {model_name_input} (version {version})")
                    

                    if metadata is None:
                        st.error("Metadata not found")
                        return
                    
                    train_sample = metadata.get("train_sample", {})

                    if not train_sample:
                        st.warning("No training sample found for drift detection")
                        train_df = None
                    else:
                        train_df = pd.DataFrame(train_sample)

                    train_cols = metadata.get("features", [])


                    if not train_cols:
                        st.error("Training schema not found in metadata")
                        return
                    
                    train_cols = metadata.get("features", [])

                    missing_cols = [
                        col for col in train_cols
                        if not (col in new_df.columns or col.endswith("_missing"))
                    ]

                    if missing_cols:

                        st.warning(
                            f"""
                            Missing columns detected:
                            {missing_cols}

                            Auto-generating placeholders.
                            """
                        )

                        for col in missing_cols:

                            # generic placeholder
                            new_df[col] = np.nan
                    
                    indicator_cols = metadata.get("indicator_cols",[])

                    for col in indicator_cols:

                        if col in new_df.columns:

                            new_df[f"{col}_missing"] = (
                                new_df[col].isnull().astype(int)
                            )
                        else:
                            new_df[f"{col}_missing"] = 1

                    cols_to_drop = metadata.get("cols_to_drop",[])

                    new_df = new_df.drop(columns=cols_to_drop,errors="ignore")


                    #Align schema (auto handled by the pipelining logic) 
                    new_df = new_df.reindex(
                    columns=train_cols,
                    fill_value=np.nan
                )                 

                    try:
                        preds = pipeline.predict(new_df)
                        

                        log_prediction(
                            model_name_input,
                            version,
                            new_df,
                            preds
                        )

                        new_df["Prediction"] = preds

                        st.success(f"Predictions using {model_name_input} (v{version})")

                        #Use safe display (avoid Arrow issue)
                        st.write("Prediction results:")
                        st.write(new_df.head().to_dict(orient="records"))

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

                    drift_detected = False
                    drift_report = {}

                    if train_df is not None:

                        #check numerical cols drift using KS drift test
                        for col in num_cols:

                            if col in new_df.columns and col in train_df.columns:

                                train_values = train_df[col].dropna()

                                new_values = new_df[col].dropna()
                                if (len(train_values) < 2 or len(new_values) < 2):
                                    continue
                                
                                stat, p_value = ks_2samp(
                                    train_values,
                                    new_values
                                )

                                drift_report[col] = {
                                    "type": "numerical",
                                    "p_value": np.round(float(p_value),5)
                                }

                                if p_value < 0.05:
                                    drift_detected = True
                        
                        #check categorical cols drift using KS drift test
                        for col in cat_cols:

                            if col in new_df.columns and col in train_df.columns:

                                train_dist = train_df[col].value_counts(normalize=True)
                                new_dist = new_df[col].value_counts(normalize=True)

                                diff = (train_dist - new_dist).fillna(0).abs().sum()

                                drift_report[col] = {
                                    "type": "categorical",
                                    "difference": np.round(float(diff),5)
                                }

                                if diff > 0.3:
                                    drift_detected = True
                        
                        st.write("### Drift Report")
                        st.write(drift_report)

                        if drift_detected:
                            st.error("Data Drift Detected — Retraining Recommended")
                        else:
                            st.success("No significant drift detected")

                        
        st.header("Training")          
        full_training_ui(y, X_train, X_test, y_train, y_test, method)

        st.divider()
        st.header("Monitoring")
        show_dashboard()

        