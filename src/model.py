"""The single canonical model and its SHAP attributions."""
import numpy as np
from catboost import CatBoostRegressor, Pool
from .config import CATBOOST, FEATURES, TARGET

def fit(train):
    m = CatBoostRegressor(**CATBOOST)
    m.fit(train[FEATURES], train[TARGET])
    return m

def shap_matrix(model, frame):
    """Signed SHAP values, one row per observation, one column per feature."""
    return model.get_feature_importance(type="ShapValues", data=Pool(frame[FEATURES]))[:, :-1]

def ranks(abs_shap):
    imp = abs_shap.mean(0)
    return (-imp).argsort().argsort() + 1, imp
