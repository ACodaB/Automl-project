from src.evaluation import evaluate_model
from sklearn.model_selection import cross_val_score
import optuna

def get_optuna_params(trial, model_name):

    if model_name == "Random Forest":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 120),
            "model__max_depth": trial.suggest_int("model__max_depth", 5, 15)
        }

    elif model_name == "XGBoost":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 120),
            "model__max_depth": trial.suggest_int("model__max_depth", 3, 8),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.2, log=True)
        }

    elif model_name == "LightGBM":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 120),
            "model__num_leaves": trial.suggest_int("model__num_leaves", 20, 60),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.2, log=True)
        }

    elif model_name == "Logistic Regression":
        return {
            "model__C": trial.suggest_float("model__C", 0.01, 10, log=True)
        }

    return {}


def optuna_objective(trial, pipeline, model_name,
                     X_tune, y_tune, problem_type):

    params = get_optuna_params(trial, model_name)
    pipeline.set_params(**params)

    if "feature_selection" in pipeline.named_steps:
        k = trial.suggest_int("feature_selection__k", 5, min(50, X_tune.shape[1]))
        pipeline.set_params(feature_selection__k=k)

    scores = cross_val_score(
        pipeline,
        X_tune,
        y_tune,
        cv=3,
        scoring="accuracy" if problem_type == "classification" else "r2"
    )

    return scores.mean()


def run_optuna(pipeline, model_name,
               X_tune, y_tune,
               problem_type, n_trials):

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        lambda trial: optuna_objective(
            trial,
            pipeline,
            model_name,
            X_tune,
            y_tune,
            problem_type
        ),
        n_trials=n_trials
    )

    return study

