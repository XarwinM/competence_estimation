import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Number of Environments  for  different Datasets
ENVS_DIC = {
    "OfficeHome": 4,
    "VLCS": 4,
    "PACS": 4,
    "TerraIncognita": 4,
    "SVIRO": 10,
    "DomainNet": 6,
}

def mix_open(features_ood, logits_ood, labels_ood,  features_open, logits_open, percentage=0.5):
    """
    Mixes features of unknown classes to existing features (open-world scneario). 
    Output contains features of unknown classes that make up {percentage} of all instances
    Arguments:
        - features_ood: Features of ood distribution
        - logits_ood: Logits of ood distribution
        - labels_ood: Labels of ood distribution
        - features_open: Features of unknown classes 
        - logits_open: Logits of unknown classes 
        - percentage: Percentage of unkowns in output
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
        - data_dir: Directory where data is saved
    Returns:
        - x_train: features on train data due to algorithm
        - logits_train: logits of classifier, i.e. predictions
        - y_train: ground truth labels on train data
        - x_val: as x_train on validation set
        - logits_val: as logits_train on validation set
        - y_val: as y_train on validation set
        - x_test: as x_train on test set
        - logits_test: as logits_train on test set
        - y_test: as y_train on test set
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