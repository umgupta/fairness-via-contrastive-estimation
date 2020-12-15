import os

import numpy
import pandas
from torchvision.datasets.utils import download_url

from lib.src.data_utils import DATA_FOLDER
from lib.src.os_utils import safe_makedirs

PCT_TRAIN = 0.8


def maybe_download():
    path = os.path.join(DATA_FOLDER, "health")
    safe_makedirs(path)
    download_url("https://raw.githubusercontent.com/ermongroup/lag-fairness/master/health.csv",
                 path)


def gather_labels(df):
    labels = []
    for j in range(df.shape[1]):
        if type(df[0, j]) is str:
            labels.append(numpy.unique(df[:, j]).tolist())
        else:
            labels.append(numpy.median(df[:, j]))
    return labels


def load_health(val_size=0.0):
    maybe_download()
    raw_df = pandas.read_csv(os.path.join(DATA_FOLDER, "health", "health.csv"))
    raw_df = raw_df[raw_df['YEAR_t'] == 'Y3']
    sex = raw_df['sexMISS'] == 0
    age = raw_df['age_MISS'] == 0
    raw_df = raw_df.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    raw_df = raw_df[sex & age]
    ages = raw_df[[f'age_{i}5' for i in range(0, 9)]]
    sexs = raw_df[['sexMALE', 'sexFEMALE']]
    charlson = raw_df['CharlsonIndexI_max']

    x = raw_df.drop(
        [f'age_{i}5' for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max',
                                              'CharlsonIndexI_min',
                                              'CharlsonIndexI_ave', 'CharlsonIndexI_range',
                                              'CharlsonIndexI_stdev',
                                              'trainset'], axis=1).to_numpy()

    labels = gather_labels(x)
    xs = numpy.zeros_like(x)
    for i in range(len(labels)):
        xs[:, i] = x[:, i] > labels[i]
    x = xs[:, numpy.nonzero(numpy.mean(xs, axis=0) > 0.05)[0]].astype(numpy.float32)

    u = ages.to_numpy().argmax(axis=1)
    y = (charlson.to_numpy() > 0).astype(numpy.float32)

    sensitive_labels = u.reshape(-1, 1)
    target_labels = y.reshape(-1, 1)
    features = x

    num_entries, num_features = features.shape
    # fixed seed to generate train and test set
    rng = numpy.random.default_rng(seed=6174)
    random_indices = numpy.arange(num_entries)
    rng.shuffle(random_indices)
    num_train_indices = int(len(random_indices) * PCT_TRAIN)
    train_indices = random_indices[:num_train_indices]
    test_indices = random_indices[num_train_indices:]

    train_data = features[train_indices]
    train_c = sensitive_labels[train_indices]
    train_labels = target_labels[train_indices]

    test_data = features[test_indices]
    test_c = sensitive_labels[test_indices]
    test_labels = target_labels[test_indices]

    if val_size != 0:
        # # shuffle
        # train_data.sample(frac=1, random_state=0)
        # train_c.sample(frac=1, random_state=0)
        # train_labels.sample(frac=1, random_state=0)

        val_cutoff = int((1 - val_size) * train_data.shape[0])
        val_data = train_data[val_cutoff:]
        train_data = train_data[:val_cutoff]

        val_labels = train_labels[val_cutoff:]
        train_labels = train_labels[:val_cutoff]

        val_c = train_c[val_cutoff:]
        train_c = train_c[:val_cutoff]

    return {"train": (
        train_data,
        train_c.astype(dtype=numpy.int),
        train_labels.astype(dtype=numpy.int),
    ), "valid": None if val_size == 0 else (
        val_data,
        val_c.astype(dtype=numpy.int),
        val_labels.astype(dtype=numpy.int),
    ), "test": (
        test_data,
        test_c.astype(dtype=numpy.int),
        test_labels.astype(dtype=numpy.int),
    )}


if __name__ == "__main__":
    data = load_health(0.2)
    print("Health dataset:")
    print(f"Train size: {data['train'][0].shape}")
    if data['valid'] is not None:
        print(f"Val size: {data['valid'][0].shape}")
    print(f"Test size: {data['test'][0].shape}")
    breakpoint()
