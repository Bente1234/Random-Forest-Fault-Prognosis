import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import TARGET

RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}
TYPE_LEVELS = ["L", "M", "H"]
RAW_NUMERIC = ["air_temp", "process_temp", "speed", "torque", "tool_wear"]
PHYSICS_NUMERIC = ["power", "temp_diff", "strain"]
FEATURE_SETS = {
    "raw": RAW_NUMERIC,
    "physics": RAW_NUMERIC + PHYSICS_NUMERIC,
}


def build_features(df):
    X = df[list(RENAME)].rename(columns=RENAME)
    X["power"] = X["torque"] * X["speed"] * 2 * np.pi / 60
    X["temp_diff"] = X["process_temp"] - X["air_temp"]
    X["strain"] = X["tool_wear"] * X["torque"]
    return X, df[TARGET]


def make_preprocessor(feature_set, scale=True):
    return ColumnTransformer(
        [
            ("type", OneHotEncoder(categories=[TYPE_LEVELS], sparse_output=False), ["type"]),
            ("numeric", StandardScaler() if scale else "passthrough", FEATURE_SETS[feature_set]),
        ],
        verbose_feature_names_out=False,
    )
