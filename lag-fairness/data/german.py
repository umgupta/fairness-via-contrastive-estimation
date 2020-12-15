import numpy as np
import pandas as pd
import tensorflow as tf

# from src.common.data.german import process_german_data

tfd = tf.contrib.distributions


def create_german_datasets(batch=64):
    # use the data we processed for laftr:
    # f_out_np = '../laftr/data/german/german.npz'
    # D = np.load(f_out_np)
    # german = ((D['x_train']).astype(np.float32), (D["attr_train"]).astype(np.float32), (D["y_train"]).astype(np.float32))
    # german_test = ((D['x_test']).astype(np.float32), (D["attr_test"]).astype(np.float32), (D["y_test"]).astype(np.float32))


    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    def transform_to_binary(df, labels):
        d = np.zeros([df.shape[0], 58])
        u = np.zeros([df.shape[0], 1])
        y = np.zeros([df.shape[0], 1])
        idx = 0
        for j in range(len(labels)):
            if type(labels[j]) is list:
                if len(labels[j]) > 2:
                    for i in range(df.shape[0]):
                        d[i, idx + int(labels[j].index(df[i, j]))] = 1
                    idx += len(labels[j])
                else:
                    for i in range(df.shape[0]):
                        d[i, idx] = int(labels[j].index(df[i, j]))
                    idx += 1
            else:
                if j != 12 and j != len(labels) - 1:
                    for i in range(df.shape[0]):
                        d[i, idx] = float(df[i, j] > labels[j])
                    idx += 1
                elif j == len(labels) - 1:
                    for i in range(df.shape[0]):
                        y[i] = float(df[i, j] > labels[j])
                else:
                    for i in range(df.shape[0]):
                        u[i] = float(df[i, j] > labels[j])
        return d.astype(np.bool), u.astype(np.bool), y.astype(np.bool)  # observation, protected, label

    d = pd.read_csv('german.data.txt', header=None, sep=' ').as_matrix()
    t = pd.read_csv('german.data.txt', header=None, sep=' ').as_matrix()
    labels = gather_labels(d)
    ds = transform_to_binary(d, labels)
    ts = transform_to_binary(t, labels)

    idx = np.arange(d.shape[0])
    np.random.seed(4)
    np.random.shuffle(idx)
    cf = int(d.shape[0] * 0.9)

    german = tuple([a[idx[:cf]].astype(np.float32) for a in ds])
    german_test = tuple([a[idx[:cf]].astype(np.float32) for a in ts])
    train = tf.data.Dataset.from_tensor_slices(german).shuffle(800).batch(batch)
    test = tf.data.Dataset.from_tensor_slices(german_test).batch(batch)
    pu = tfd.Bernoulli(probs=np.mean(german[1]))
    return train, test, pu