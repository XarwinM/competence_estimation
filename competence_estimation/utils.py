import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#ys.path.append("/home/tarkus/Desktop/WILDS/paper_2022/surprised_classifiers")
#from surprised_classifiers.models.inns import ConditionalINN

def mix_open(features_ood, logits_ood, labels_ood,  features_open, logits_open, percentage=0.5):
    """
    Mixes features of unknown classes to existing features. 
    Output contains features of unknown classes that make up {percentage}
    Arguments:
        - features_ood: Features of ood distribution
        - logits_ood: Logits of ood distribution
        - labels_ood: Labels of ood distribution
        - features_open: Features of unknown classes 
        - logits_open: Logits of unknown classes 
        - percentage: Percentage of unkow
    Returns:
        - features_out: Features mixed with unkown classes
        - logits_out: Logits mixed with unknown classes
        - labels_out: Labels of known classes and label -1 for unknown class

    """
    n = features_ood.shape[0]
    n_open= int(n*percentage / (1-percentage))
    if n_open <= features_open.shape[0]:
        pass
    else:
        # If we cannot achieve the right percentage with n samples of known classes
        # In this case we have to 'shorten' n
        n = int(features_open.shape[0]* ((1-percentage)/percentage))
    torch.manual_seed(0)
    idx = torch.randperm(features_open.shape[0])
    features_open = features_open[idx]

    features_out = np.concatenate( ( features_ood[:n], features_open[:n_open]))
    logits_out = np.concatenate( (logits_ood[:n], logits_open[:n_open]))

    labels_out = np.concatenate((labels_ood[:n], np.zeros((logits_open[:n_open].shape[0])) -1), 0)

    assert labels_out.shape[0] * percentage - min(n_open, logits_open.shape[0]) >= -1 
    assert labels_out.shape[0] * percentage - min(n_open, logits_open.shape[0]) <= 1 

    return features_out, logits_out, labels_out

def get_network_weights(algorithm, dataset, test_domain, dataset_path):
    """
    Get Network weights of trained model
    """
    W = np.load(f"{dataset_path}/{dataset}/test_env_{test_domain}/{algorithm}_W.npy")
    b = np.load(f"{dataset_path}/{dataset}/test_env_{test_domain}/{algorithm}_b.npy")
                
    return W, b

def load_data(algorithm, dataset, test_domain, data_dir, fast=False):
    """
    Loads train, validation and test data. Train and validation data follow the same distribution
    Arguments:
        - algorithm: Classifier Algorithm, e.g. ERM
        - dataset: dataset we consider, e.g. RotatedMNIST
        - test_domain: Domain that is considered to be the test domain
        - data_dir: Director where data is saved
    Returns:
        - x_train: features on train data due to algorithm
        - logits_train: logits of classifier, i.e. predictions
        - y_train: ground truth labels on train data
        _ x_val: as x_train on validation set
        _ logits_val: as logits_train on validation set
        _ y_val: as y_train on validation set
        _ x_test: as x_train on test set
        _ logits_test: as logits_train on test set
        _ y_test: as y_train on test set
    """
    dataset_path = f"{data_dir}/{dataset}/test_env_{test_domain}/"
    
    x_iid_train, y_iid_train = np.load(f"{dataset_path}/{algorithm}_features_iid_train.npy") , np.load(f"{dataset_path}/{algorithm}_labels_iid_train.npy")
    x_iid_val, y_iid_val = np.load(f"{dataset_path}/{algorithm}_features_iid_val.npy"), np.load(f"{dataset_path}/{algorithm}_labels_iid_val.npy")
    x_iid_test, y_iid_test =  np.load(f"{dataset_path}/{algorithm}_features_iid_test.npy") , np.load(f"{dataset_path}/{algorithm}_labels_iid_test.npy")
    x_ood_test, y_ood_test =  np.load(f"{dataset_path}/{algorithm}_features_ood_test.npy"), np.load(f"{dataset_path}/{algorithm}_labels_ood_test.npy")

    logits_iid_train = np.load(f"{dataset_path}/{algorithm}_logits_iid_train.npy")
    logits_iid_val = np.load(f"{dataset_path}/{algorithm}_logits_iid_val.npy")
    logits_iid_test = np.load(f"{dataset_path}/{algorithm}_logits_iid_test.npy")
    logits_ood_test = np.load(f"{dataset_path}/{algorithm}_logits_ood_test.npy")

    return (
        (x_iid_train, logits_iid_train, y_iid_train),
        (x_iid_val, logits_iid_val, y_iid_val),
        (x_iid_test, logits_iid_test, y_iid_test),
        (x_ood_test, logits_ood_test, y_ood_test),
    )
def load_model_from_path(model_name, data_dir, model_dir):

    dataset = re.search("cinn_(.*)_domain", model_name).group(1)
    test_domain = re.search(f"domain(.*)_algorithm", model_name).group(1)
    algorithm = re.search(f"algorithm(.*)_lr", model_name).group(1)


    hyperparameters = {}
    keywords = {
        "lr": "lr",
        "bs": "batch_size",
        "coulingb": "n_coupling_blocks",
        "subnet_dim": "subnet_dim",
        "noise": "noise_scale",
        "patience": "patience",
    }

    for e, k in enumerate(keywords.keys()):
        if e < len(keywords) - 1:
            hyperparameters[keywords[k]] = re.search(f"_{k}(.*)_{list(keywords.keys())[e+1]}", model_name).group(1)
        else:
            hyperparameters[keywords[k]] = re.search(f"_{k}(.*).pt", model_name).group(1)

    hyperparameters['latent_dim'] = 512
    hyperparameters['n_classes'] = 6
    hyperparameters['p'] = 0.5

    return load_model(algorithm, dataset, test_domain, model_dir, data_dir, **hyperparameters)


def _gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes the Gaussian Kernel matrix
    """
    dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)[:, :, None]
    beta = 1.0 / (2.0 * torch.Tensor(sigmas)[None, :]).cuda()
    s = torch.matmul(dist, beta)
    k = torch.exp(-s).sum(axis=-1)
    return k


def _mmd(x, y, factor_z=1):
    """
    Computes the Maximum-Mean-Discrepancy (MMD): MMD(embedding,z) where
    z follows a standard normal distribution
    Computation is performed for multiple scales
    """
    # z = torch.randn(embedding.shape)
    #z = torch.randn(embedding.shape[0] * factor_z, embedding.shape[1]).cuda()
    sigmas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]

    loss = torch.mean(_gaussian_kernel_matrix(x, x, sigmas))
    loss += torch.mean(_gaussian_kernel_matrix(y, y, sigmas))
    loss -= 2 * torch.mean(_gaussian_kernel_matrix(x, y, sigmas))
    return loss

class _ECELoss(nn.Module):
    """
    due to https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
