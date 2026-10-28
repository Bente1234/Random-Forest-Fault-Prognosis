from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import SEED
from features import make_preprocessor

MODEL_SPECS = {
    "dummy": {
        "estimator": DummyClassifier,
        "fixed": {"strategy": "prior"},
        "grid": {},
        "scale": False,
        "complexity": 0,
    },
    "logistic_regression": {
        "estimator": LogisticRegression,
        "fixed": {"max_iter": 5000},
        "grid": {"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]},
        "scale": True,
        "complexity": 1,
    },
    "decision_tree": {
        "estimator": DecisionTreeClassifier,
        "fixed": {"random_state": SEED},
        "grid": {"max_depth": [2, 3, 4, 5], "min_samples_leaf": [1, 10], "class_weight": [None, "balanced"]},
        "scale": False,
        "complexity": 2,
    },
    "random_forest": {
        "estimator": RandomForestClassifier,
        "fixed": {"n_estimators": 200, "random_state": SEED, "n_jobs": 1},
        "grid": {"max_depth": [None, 8], "min_samples_leaf": [1, 5], "class_weight": [None, "balanced_subsample"]},
        "scale": False,
        "complexity": 3,
    },
}
FEATURE_SET_COMPLEXITY = {"raw": 0, "physics": 1}
CANDIDATES = [
    (model, feature_set)
    for model in MODEL_SPECS
    for feature_set in FEATURE_SET_COMPLEXITY
    if not (model == "dummy" and feature_set == "physics")
]


def make_pipeline(model, feature_set, **params):
    spec = MODEL_SPECS[model]
    estimator = spec["estimator"](**spec["fixed"], **params)
    return Pipeline([
        ("preprocess", make_preprocessor(feature_set, scale=spec["scale"])),
        ("model", estimator),
    ])


def param_grid(model):
    return {f"model__{name}": values for name, values in MODEL_SPECS[model]["grid"].items()}


def complexity(model, feature_set):
    return MODEL_SPECS[model]["complexity"], FEATURE_SET_COMPLEXITY[feature_set]
