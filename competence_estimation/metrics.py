"""
This module provides functions to compute various interesting metrics of the Safe Curve G(alpha)
"""

import torch
import numpy as np


def compute_metric(
    score_id_val,
    score_id_test,
    score_ood_test,
    logits_id_val,
    labels_id_val,
    logits_id_test,
    labels_id_test,
    logits_ood_test=None,
    labels_ood_test=None,
    metrics=["quantile_95"],
):
    """
    Returns a dictionary of different queried metrics
    Arguments:
        - scores_id_val: numpy array of scores on in distribution data (validation set)
        - scores_id_test: numpy array of scores on in distribution data (test set)
        - scores_ood_test: numpy array of scores on ood data
        - data_id_val: input data from training distribution (i.e. features of classifier)
        - logits_id_val: logits of classifier on id validation set
        - labels_id_val: ground truth labels on id validation set
        - data_id_test: input data from id test distribution (i.e. features of classifier)
        - logits_id_test: logits of classifier on id test set
        - labels_id_test: ground truth labels on id id set
        - data_ood_test: input data from test distribution (i.e. features of classifier)
        - logits_ood_test: logits of classifier on data_test
        - labels_ood_test: ground truth labels for data_test
        - metrics: list of metrics that evaluate safe region
        - num_alphas: amount of quantiles on which metrics are computed
        - ece_bins: number of bins to compute ece
    Return:
        - out: dictionary with (a) name of metric as key and (b) computed value of this metric
    """

    if logits_ood_test is None or labels_ood_test is None:
        ood_prediction = False
    else:
        ood_prediction = True

    possible_metrics = [
        "accuracy",  # accuracy on data sets
        "quantile_95",  # 95 percentile
    ]
    assert all(metric in possible_metrics for metric in metrics)

    # Compute true-false values on id val data
    true_false_id_val = (
        (torch.from_numpy(logits_id_val).argmax(1) == torch.from_numpy(labels_id_val))
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )

    # Compute true-false values on id test data
    true_false_id_test = (
        (torch.from_numpy(logits_id_test).argmax(1) == torch.from_numpy(labels_id_test))
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )

    # Compute accuracy on validation, test and ood data
    accuracy_id_val = np.sum(true_false_id_val) / true_false_id_val.shape[0]
    accuracy_id_test = np.sum(true_false_id_test) / true_false_id_test.shape[0]

    if ood_prediction:
        # Compute true-false values on ood test data
        true_false_ood_test = (
            (
                torch.from_numpy(logits_ood_test).argmax(1)
                == torch.from_numpy(labels_ood_test)
            )
            .float()
            .cpu()
            .view(-1)
            .numpy()
        )
        accuracy_ood_test = np.sum(true_false_ood_test) / true_false_ood_test.shape[0]

    # Output dictionary
    out = {}

    if "accuracy" in metrics:
        out["acc_id_test"] = accuracy_id_test
        out["acc_id_val"] = accuracy_id_val
        if ood_prediction:
            out["acc_ood_test"] = accuracy_ood_test

    # Compute 95 percentile where the 95 percentile is defined on the id val data
    if "quantile_95" in metrics:
        qs = np.quantile(score_id_val, 0.95)

        if ood_prediction:
            mask = torch.from_numpy(score_ood_test) < qs
            out["n_95_frac_ood"] = mask.sum().item() / logits_ood_test.shape[0]
            out["n_95_ood"] = (
                (
                    torch.from_numpy(logits_ood_test[mask]).argmax(1)
                    == torch.from_numpy(labels_ood_test[mask])
                ).sum()
                / mask.sum()
            ).item()

        mask = torch.from_numpy(score_id_test) < qs
        out["n_95_frac_id_test"] = mask.sum().item() / logits_id_test.shape[0]
        out["n_95_id_test"] = (
            (
                torch.from_numpy(logits_id_test[mask]).argmax(1)
                == torch.from_numpy(labels_id_test[mask])
            ).sum()
            / mask.sum()
        ).item()

    return out
