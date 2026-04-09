from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

def get_models(problem_type):

    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),

            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

            "SVM": SVC(probability=True),

            "XGBoost": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),

            "LightGBM": LGBMClassifier(
                random_state=42,
                n_jobs=-1
            )
        }

    else:
        return {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

            "SVR": SVR(),

            "XGBoost": XGBRegressor(
                random_state=42,
                n_jobs=-1
            ),

            "LightGBM": LGBMRegressor(
                random_state=42,
                n_jobs=-1
            )
        }
    

def recommend_models(n_samples,n_features):
    

    if n_samples < 10000:
        rec = ["XGBoost", "Random Forest"]
        if n_features < 50:
            rec.append("SVM")
        return rec

    elif n_samples < 100000:
        return ["LightGBM", "XGBoost", "Random Forest"]

    else:
        rec = ["LightGBM"]
        if n_features < 100:
            rec.append("Random Forest")
        return rec


def filter_models(n_samples, models):
    

    if n_samples > 50000:
        models = [m for m in models if m not in ["SVM", "SVR"]]

    if n_samples > 200000:
        models = [m for m in models if m != "Random Forest"]

    return models