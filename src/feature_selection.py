from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectFromModel


def supports_feature_importance(model):
    return hasattr(model, "feature_importances_") or hasattr(model, "coef_")

def get_feature_selector(method, problem_type=None, model=None,k=None):

    if method == "none":
        return None

    # ------------------------
    # FILTER METHOD
    # ------------------------
    elif method == "filter":

        score_func = f_classif if problem_type == "classification" else f_regression

        return SelectKBest(score_func=score_func, k=k if k is not None else "all")

    # ------------------------
    # MODEL-BASED
    # ------------------------
    elif method == "model":

        if model is None:
            raise ValueError("Model required for model-based feature selection")
        
        if not supports_feature_importance(model):
            return None
        return SelectFromModel(model)


    # ------------------------
    # HYBRID
    # ------------------------
    elif method == "hybrid":

        score_func = f_classif if problem_type == "classification" else f_regression

        if model is None:
            raise ValueError("Model required for hybrid feature selection")

        if not supports_feature_importance(model):
            return SelectKBest(score_func=score_func, k="all")

        return [
            ("filter", SelectKBest(score_func=score_func, k=k if k is not None else "all")),
            ("model_select", SelectFromModel(model))
        ]

    else:
        raise ValueError("Invalid feature selection method")