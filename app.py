import streamlit as st
from src.data import load_data,split_data
from src.util import (detect_problem_type, results_to_df,detect_columns,
    suggest_target,data_report)
from src.training import run_experiments
from src.models import get_models, filter_models, recommend_models
from src.registry import (load_model_metadata,
                          get_registered_models,load_model_by_dataset)
import hashlib

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



        report = data_report(df)
        st.subheader("Dataset Summary")
        st.write("Shape:", report["Data shape"])
        st.write("Missing Values:", report["Missing values"])
        st.write("Data Types:", report["Column datatype"])



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



                if st.session_state.results:

                    df_results = results_to_df(st.session_state.results)

                    st.subheader("Results")
                    st.dataframe(df_results)

                    if problem_type == "classification":
                        best = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
                    else:
                        best = df_results.sort_values(by="R2 Score", ascending=False).iloc[0]

                    st.success(f"Best Model: {best['Model']}")

                    
                    deployment_ui(selected_models, target)
                    # safe_df=make_streamlit_safe(df_results)
                
                
        def deployment_ui(selected_models, target):

    
            st.subheader("Make Predictions")

            #get models from registry
            registered_models = get_registered_models()

            valid_models=[]

            # 🔹 filter based on dataset_id
            model,best_version=load_model_by_dataset()

            #get models from session
            session_models = st.session_state.get("trained_models", [])

            #combine both (no duplicates)
            available_models = list(set(valid_models + session_models))

            with st.expander("Model Info"):
                st.write("Registered Models:", registered_models)
                st.write("Session Models:", session_models)

            if not available_models:
                st.warning("No models available. Train a model first.")
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

                    model, version = load_model_by_dataset(model_name_input,dataset_id)

                    if model is None:
                        st.error("No saved model found")
                        return
                    st.info(f"Using model: {model_name_input} (version {version})")
                    
                    metadata = load_model_metadata(model_name_input, version)

                    if metadata is None:
                        st.error("Metadata not found")
                        return

                    train_cols = metadata.get("features", [])

                    if not train_cols:
                        st.error("Training schema not found in metadata")
                        return
                    
                    missing_cols = [col for col in train_cols if col not in new_df.columns]
                    if missing_cols:
                        st.warning(f"Missing columns detected: {missing_cols}")


                    #Align schema (CRITICAL STEP)
                    new_df_aligned = new_df.reindex(columns=train_cols, fill_value=0)

                    try:
                        preds = model.predict(new_df_aligned)

                        new_df["Prediction"] = preds

                        st.success(f"Predictions using {model_name_input} (v{version})")

                        #Use safe display (avoid Arrow issue)
                        st.write("Prediction results: ")
                        st.write(new_df.head().to_dict(orient="records"))

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
        full_training_ui(y, X_train, X_test, y_train, y_test, method)

        