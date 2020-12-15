import logging

import torch
from box import Box
from torch.utils.data import TensorDataset

from src.common.data.adult import load_adult
from src.common.data.health import load_health

logger = logging.getLogger()


def get_dataset(data_config, device="cpu"):
    """ Take config and return dataloader"""
    dataname = data_config.name
    val_size = data_config.val_size

    if dataname == "adult":
        data = load_adult(val_size=val_size)
        c_size = 2
        c_type = "binary"
    elif dataname == "health":
        data = load_health(val_size=val_size)
        c_size = 9
        c_type = "one_hot"
    else:
        logger.error(f"Invalid data name {dataname} specified")
        raise Exception(f"Invalid data name {dataname} specified")

    train, valid, test = data["train"], data["valid"], data["test"]
    if valid is None:
        valid = data["test"]

    return (
        Box({"train": TensorDataset(
            torch.tensor(train[0]).float().to(device),
            torch.tensor(train[1]).long().to(device),
            torch.tensor(train[2]).long().to(device),
        ), "test": TensorDataset(
            torch.tensor(test[0]).float().to(device),
            torch.tensor(test[1]).long().to(device),
            torch.tensor(test[2]).long().to(device),
        ), "valid": TensorDataset(
            torch.tensor(valid[0]).float().to(device),
            torch.tensor(valid[1]).long().to(device),
            torch.tensor(valid[2]).long().to(device),
        )}), {
            "input_shape": train[0].shape[1:],
            "c_size": c_size,
            "c_type": c_type,
            "y_size": 2,
            "y_type": "binary",
        })
