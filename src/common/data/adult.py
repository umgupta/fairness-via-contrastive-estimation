import os

import numpy
import pandas
from torchvision.datasets.utils import download_url

from lib.src.data_utils import DATA_FOLDER
from lib.src.os_utils import safe_makedirs

ADULT_RAW_COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                       "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                       "hours-per-week", "native-country", "income", ]
# this are categorical column indices; doesnot include sex and income column (these are binary)
ADULT_RAW_COL_FACTOR = [1, 3, 5, 6, 7, 8, 13]


def maybe_download():
    path = os.path.join(DATA_FOLDER, "adult")
    safe_makedirs(path)
    download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", path)
    download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", path)


def load_adult(val_size=0.0):
    maybe_download()
    # taken from https://github.com/dcmoyer/inv-rep/blob/master/src/uci_data.py

    # load data as pandas table
    train_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.data"), delimiter=", ",
                                   header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                   keep_default_na=False)
    test_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.test"), delimiter=", ",
                                  header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                  keep_default_na=False, skiprows=1)

    # drop missing val
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # concatenate and binarize categorical variables
    all_data = pandas.concat([train_data, test_data])
    all_data = pandas.get_dummies(all_data,
                                  columns=[ADULT_RAW_COL_NAMES[i] for i in ADULT_RAW_COL_FACTOR])
    # fix binary variables now
    all_data.loc[all_data.income == ">50K", "income"] = 1
    all_data.loc[all_data.income == ">50K.", "income"] = 1
    all_data.loc[all_data.income == "<=50K", "income"] = 0
    all_data.loc[all_data.income == "<=50K.", "income"] = 0

    all_data.loc[all_data.sex == "Female", "sex"] = 0
    all_data.loc[all_data.sex == "Male", "sex"] = 1
    cutoff = train_data.shape[0]

    all_x = all_data.iloc[:, (all_data.columns != "income") & (all_data.columns != "sex")]
    all_c = all_data.iloc[:, all_data.columns == "sex"]
    all_labels = all_data.iloc[:, all_data.columns == "income"]

    # col_valid = [len(all_x.iloc[:, all_x.columns==x].unique()) > 1 for x in all_x.columns]
    # all_x = all_x.iloc[:, col_valid]

    # normalization
    maxes = all_x.max(axis=0)
    all_x = all_x / maxes

    train_data = all_x[:cutoff]
    train_c = all_c[:cutoff]
    train_labels = all_labels[:cutoff]

    test_data = all_x[cutoff:]
    test_c = all_c[cutoff:]
    test_labels = all_labels[cutoff:]

    # split off validation data (we keep val_size for training 0, so this code is never run)
    if val_size != 0:
        # # shuffle
        # train_data.sample(frac=1, random_state=0)
        # train_c.sample(frac=1, random_state=0)
        # train_labels.sample(frac=1, random_state=0)

        val_cutoff = int((1 - val_size) * train_data.shape[0])

        val_data = train_data.iloc[val_cutoff:, :]
        train_data = train_data.iloc[:val_cutoff, :]

        val_labels = train_labels.iloc[val_cutoff:, :]
        train_labels = train_labels.iloc[:val_cutoff, :]

        val_c = train_c.iloc[val_cutoff:, :]
        train_c = train_c.iloc[:val_cutoff, :]

    return {"train": (
        train_data.to_numpy(),
        train_c.to_numpy(dtype=numpy.int),
        train_labels.to_numpy(dtype=numpy.int),
    ), "valid": None if val_size == 0 else (
        val_data.to_numpy(),
        val_c.to_numpy(dtype=numpy.int),
        val_labels.to_numpy(dtype=numpy.int),
    ), "test": (
        test_data.to_numpy(),
        test_c.to_numpy(dtype=numpy.int),
        test_labels.to_numpy(dtype=numpy.int),
    )}


if __name__ == "__main__":
    data = load_adult(0.2)
    print("Adult dataset:")
    print(f"Train size: {data['train'][0].shape}")
    if data['valid'] is not None:
        print(f"Val size: {data['valid'][0].shape}")
    print(f"Test size: {data['test'][0].shape}")
