import pandas as pd

from config import CLEAN_DATA, TRAIN_IDS, TEST_IDS, TARGET, FAILURE_MODES


def failure_stratum(df):
    first_mode = df[FAILURE_MODES].idxmax(axis=1)
    return first_mode.where(df[TARGET] == 1, "none")


def load_train_test():
    df = pd.read_csv(CLEAN_DATA)
    train_ids = pd.read_csv(TRAIN_IDS)["UDI"]
    test_ids = pd.read_csv(TEST_IDS)["UDI"]
    train = df[df["UDI"].isin(train_ids)].reset_index(drop=True)
    test = df[df["UDI"].isin(test_ids)].reset_index(drop=True)
    return train, test
