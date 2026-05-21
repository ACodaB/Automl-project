from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import skew
from src.feature_selection import get_feature_selector
import numpy as np

def drop_high_missing_col(X,threshold=0.5):
    missing_ratio = X.isnull().mean()

    cols_to_drop = missing_ratio[
        missing_ratio > threshold
    ].index

    X = X.drop(columns=cols_to_drop)

    return X, cols_to_drop


def get_num_impute_strategy(df, num_cols):
    strategies = {}

    for col in num_cols:
        col_data = df[col].dropna()

        if len(col_data) == 0:
            strategies[col] = "mean"
            continue

        skewness = skew(col_data)

        if abs(skewness) < 0.5:
            strategies[col] = "mean"
        else:
            strategies[col] = "median"

    return strategies

def build_preprocessor(X,num_cols,cat_cols):

    X,cols_to_drop=drop_high_missing_col(X)

    # update numerical cols
    num_cols = [
        col for col in num_cols
        if col not in cols_to_drop
    ]

    # update categorical cols
    cat_cols = [
        col for col in cat_cols
        if col not in cols_to_drop
    ]

    indicator_cols=[]
    # Add missing indicators
    for col in num_cols:

        missing_ratio = X[col].isnull().mean()

        # Add indicator for moderate missingness(10-50%) for MAR and MNAR
        if 0.1 <= missing_ratio <= 0.5:

            X[f"{col}_missing"] = (
                X[col].isnull().astype(int)
            )
            indicator_cols.append(col)
        
            # Refresh numerical columns
            num_cols = X.select_dtypes(
                include=np.number
            ).columns.tolist()


     # Get strategies
    strategies = get_num_impute_strategy(X, num_cols)

    # ---------------------------
    # NUMERICAL PIPELINE
    # ---------------------------
    num_pipelines = []
    for col in num_cols:

        # missing_ratio = X[col].isnull().mean()

        # # Moderate missingness
        # if 0.1 <= missing_ratio <= 0.5:

        #     pipeline = Pipeline([
        #         ("imputer", KNNImputer(n_neighbors=5)),
        #         ("scaler", StandardScaler())
        #     ])

        # # Low missingness(<10%)
        # else:

        pipeline = Pipeline([
            ("imputer",
            SimpleImputer(strategy=strategies[col])),
            ("scaler", StandardScaler())
        ])

    num_pipelines.append((col, pipeline, [col]))

    # ---------------------------
    # CATEGORICAL PIPELINE
    # ---------------------------
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant",fill_value="missing")),  # fill missing
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))    # encode
    ])

    # ---------------------------
    # COMBINE BOTH
    # ---------------------------
    transformers=num_pipelines+[("cat", cat_pipeline, cat_cols)]
    preprocessor = ColumnTransformer(transformers)

    return (preprocessor,X,cols_to_drop,indicator_cols)

def build_pipeline(preprocessor, model, feature_method="none", problem_type=None):

    steps = [
        ("preprocessing", preprocessor)
    ]

    selector = get_feature_selector(
        feature_method,
        problem_type=problem_type,
        model=model
    )

    if isinstance(selector, list):
        steps.extend(selector)

    elif selector is not None:
        steps.append(("feature_selection", selector))

    steps.append(("model", model))

    return Pipeline(steps)