from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import skew
from src.feature_selection import get_feature_selector
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

     # Get strategies
    strategies = get_num_impute_strategy(X, num_cols)

    # ---------------------------
    # NUMERICAL PIPELINE
    # ---------------------------
    num_pipelines = []
    for col in num_cols:
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=strategies[col])),
            ("scaler", StandardScaler())
        ])
        num_pipelines.append((col, pipeline, [col]))
    # ---------------------------
    # CATEGORICAL PIPELINE
    # ---------------------------
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))    # encode
    ])

    # ---------------------------
    # COMBINE BOTH
    # ---------------------------
    transformers=num_pipelines+[("cat", cat_pipeline, cat_cols)]
    preprocessor = ColumnTransformer(transformers)

    return preprocessor

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