import numpy as np
import pandas as pd
def detect_columns(df):
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    return num_cols, cat_cols

def suggest_target(df):
    # heuristic: column with low unique values (classification)
    for col in df.columns:
        if df[col].nunique() < 10:
            return col
    return df.columns[-1]

def data_report(df): 
    return {"Missing values": df.isnull().sum().to_dict(),
    "Data shape": df.shape,
    "Column datatype":df.dtypes.astype(str).to_dict()
    }



def detect_problem_type(y):
    
    # 1. Object → classification
    if y.dtype == "object":
        return "classification"
        

    unique_vals = y.nunique()
    total_vals = len(y)
    unique_ratio = unique_vals / total_vals

    # 2. Strong discrete signal
    if unique_vals <= 10:
        return "classification"

    # 3. sqrt heuristic
    if unique_vals < np.sqrt(total_vals):
        return "classification"

    # 4. ratio check
    if unique_ratio < 0.02:
        return "classification"

    # 5. integer type with moderate uniques
    if pd.api.types.is_integer_dtype(y) and unique_vals < 50:
        return "classification"

    return "regression"

# def make_streamlit_safe(df):
    
#     df = df.copy()

   
#     for col in df.columns:
#         df[col] = df[col].apply(lambda x:
#             float(x) if isinstance(x, (np.integer, np.floating))
#             else str(x)
#         )

#     return df


def results_to_df(results):

    rows = []

    for r in results:
        row = {
            "Run ID": str(r["run_id"]),
            "Model": str(r["model"]),
            "Time (s)": float(r["training_time"])
        }

        for key,val in r["metrics"].items():
            row[key]=float(val) if val is not None else None

        
        
        rows.append(row)

   
    return pd.DataFrame(rows)

    