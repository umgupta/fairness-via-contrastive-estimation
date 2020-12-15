import numpy as np

from src.common.data.german import process_german_data

if __name__=="__main__":
    data  = process_german_data(val_size=0)
    f_out_np = 'laftr/data/german/german.npz'
    train = data["train"]
    test = data["test"]

    D = {"training": {}, "test":{}}
    D["training"]["X"] = train[0]
    D["training"]["Y"] = train[2]
    D["training"]["A"] = train[1]

    D["test"]["X"] = test[0]
    D["test"]["Y"] = test[2]
    D["test"]["A"] = test[1]

    # since we don't want to use the validation strategy we can reduce this to minimum so that all the training data is used. But this shouldnot matter much
    n = D['training']['X'].shape[0]
    shuf = np.random.permutation(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = shuf[:valid_ct]
    train_inds = shuf[valid_ct:]

    np.savez(f_out_np, x_train=D['training']['X'], x_test=D['test']['X'],
        y_train=D['training']['Y'], y_test=D['test']['Y'],
        attr_train=D['training']['A'], attr_test=D['test']['A'],
        train_inds=train_inds, valid_inds=valid_inds)
