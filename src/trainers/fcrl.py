import logging
import math

import numpy as np
import torch
from box import Box
from sklearn.metrics import roc_auc_score

from lib import BaseTrainer
from src.common import metrics
from src.common.viz import do_classification

logger = logging.getLogger()


class FCRLTrainer(BaseTrainer):

    def compute_encoding(self, loader):
        # This function expects a trainer, and data loader and extracts representations for CVIB,
        # and other FCRL like models
        # return dict of np array

        n = loader.dataset.tensors[0].shape[0]

        mu = np.zeros((n, self.model.z_size))
        sigma = np.zeros((n, self.model.z_size))
        y = np.zeros(n)
        dim_c = math.ceil(math.log2(self.model.c_size)) if self.model.c_type == "binary" else 1

        c = np.zeros((n, dim_c))
        # NOTE: Only working for binary and one_hot encoding
        z = np.zeros((n, self.model.z_size))

        batch_size = loader.batch_size
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x_ = batch[0].to(self.model.device)
                c_ = batch[1].to(self.model.device)

                temp = self.model.forward(batch)
                f, s = temp["f"], temp["s"]
                mu[i * batch_size: (i + 1) * batch_size] = f.cpu().data.numpy()
                sigma[i * batch_size: (i + 1) * batch_size] = s.cpu().data.numpy()

                # batch has y labels
                if len(batch) > 2:
                    y[i * batch_size: (i + 1) * batch_size] = batch[2].view(-1).cpu().data.numpy()
                c[i * batch_size: (i + 1) * batch_size] = c_.cpu().data.numpy().reshape(-1, dim_c)
                z[i * batch_size: (i + 1) * batch_size] = self.model._sample(f,
                                                                             s).cpu().data.numpy()

        result = Box({"mu": mu, "sigma": sigma, "z": z, "c": c})

        if len(batch) > 2:
            result["y"] = y

        return result

    def validate(self, train_loader, valid_loader, *args, **kwargs):
        loss, aux_loss = super().validate(train_loader, valid_loader)
        return loss, aux_loss

    def test(self, train_loader, test_loader, *args, **kwargs):
        # here we will compute stats for both train and test data

        logger.info("Computing loss stats for train data")
        train_loss, train_aux_loss = super().validate(self, train_loader, train_loader)

        logger.info("Computing loss stats for test data")
        loss, aux_loss = super().validate(self, train_loader, test_loader)

        logger.info("Computing encoding for train data")
        temp = self.compute_encoding(train_loader)
        mu_train, sigma_train, z_train, c_train = temp.mu, temp.sigma, temp.z, temp.c
        y_train = temp.get("y")

        logger.info("Computing encoding for test data")
        temp = self.compute_encoding(test_loader)
        mu_test, sigma_test, z_test, c_test = temp.mu, temp.sigma, temp.z, temp.c
        y_test = temp.get("y")

        logger.info("Saving encoding")
        np.save(f"{self.result_dir}/embedding.npy", {"mu_train": mu_train,
                                                     "mu_test": mu_test,
                                                     "sigma_train": sigma_train,
                                                     "sigma_test": sigma_test,
                                                     "z_train": z_train,
                                                     # store it just in case we need to reproduce
                                                     "z_test": z_test,
                                                     # store it just in case we need to reproduce
                                                     "c_train": c_train,
                                                     "c_test": c_test,
                                                     "y_train": y_train,
                                                     "y_test": y_test,
                                                     })

        # train a classifier on representation and compute accuracy
        if y_train is not None and y_test is not None:
            # samples
            logger.info("Train classifier on representation")
            # random forest
            logger.info("Random forest classifier")
            score, rf_prob = do_classification(z_train, y_train, z_test, y_test, simple=True)
            aux_loss.update({"rf_acc_sample": score})

            # neural network
            logger.info("Two layer neural network classifier")
            score, nn_prob = do_classification(z_train, y_train, z_test, y_test, simple=False)
            aux_loss.update({"nn_acc_sample": score})

            if self.model.y_type == "binary":
                dp_nn_s, _ = metrics.demographic_parity_difference_soft(y_test, c_test, nn_prob)
                aux_loss.update({"demographic_parity_nn_sample_soft": dp_nn_s})

                dp_nn_s, _ = metrics.demographic_parity_difference(y_test, c_test, nn_prob)
                aux_loss.update({"demographic_parity_nn_sample": dp_nn_s})

                dp_rf_s, _ = metrics.demographic_parity_difference_soft(y_test, c_test, rf_prob)
                aux_loss.update({"demographic_parity_rf_sample_soft": dp_rf_s})

                dp_rf_s, _ = metrics.demographic_parity_difference(y_test, c_test, rf_prob)
                aux_loss.update({"demographic_parity_rf_sample": dp_rf_s})

                nn_auc_s = roc_auc_score(y_test, nn_prob[:, 1])
                rf_auc_s = roc_auc_score(y_test, rf_prob[:, 1])
                aux_loss.update({"nn_auc_sample": nn_auc_s, "rf_auc_sample": rf_auc_s})
            else:
                raise Exception("reminder: y_type is not binary and we need to check this")

        return loss, aux_loss
