"""
This module provides functions to compute various interesting metrics of the Safe Curve G(alpha)
"""

import os
from prettytable import PrettyTable
import re
import pickle
from types import SimpleNamespace

import click
from sklearn.metrics import auc
import torch
import numpy as np

# from surprised_classifiers.utils import model_selection, load_data, load_model_from_path
from competence_estimation.utils import load_data, _ECELoss


def compute_curves(
    scores_iid,
    scores_ood,
    logits_ood_test,
    labels_ood_test,
    num_alphas=500,
    alpha_start=0.000,
    alpha_end=1.0,
    metric='acc',
    ece_bins=10,
):
    """
    Compute metric-curves (accuracy or empirical calibration error) on safe region S_alpha as well as fraction of S_alpha w.r.t test set
    Arguments:
        - scores_iid: numpy array of suprisals on training/validation set
        - scores_ood: numpy array surprisals on test set
        - logits_ood_test: logits on ood test set corresponding to scores_ood
        - labels_ood_test: ground truth labels on ood test set corresponding to logits_ood_test
        - num_alphas: number of alpha values we consider
        - alpha_start: starting range of alpha values
        - alpha_end: highest alpha value we consider
        - metric: acc (accuracy) or ece (empirical calibratoin error)
    Returns:
        - x_axis: alpha values where |S_alpha| > 0
        - metric_alpha: accuracy or ece in Safe Region S_alpha for alphas in x_axis
        - fracs: fraction of samples in safe rgeion S_alpha
    """
    if metric == 'acc':
        true_false_test = (
            (torch.from_numpy(logits_ood_test).argmax(1) == torch.from_numpy(labels_ood_test))
            .float()
            .cpu()
            .view(-1)
            .numpy()
        )
    elif metric == 'ece':
        ece = _ECELoss(n_bins=ece_bins)

    alphas = np.linspace(alpha_start, alpha_end, num_alphas)
    qs = np.quantile(scores_iid, alphas)

    metric_alpha = []
    fracs = []

    # x_axis is the domain of G(alpha), i.e. where |S_alpha| > 0
    x_axis = []
    for e, q in enumerate(qs):
        if metric =='acc':
            remaining = true_false_test[scores_ood <= q]
            if remaining.shape[0] > 0:
                acc = np.sum(remaining) / remaining.shape[0]
                frac = remaining.shape[0] / true_false_test.shape[0]
                metric_alpha.append(acc)
                fracs.append(frac)
                x_axis.append(alphas[e])
        if metric == 'ece':
            remaining_labels = labels_ood_test[scores_ood <= q]
            remaining_logits = logits_ood_test[scores_ood <= q]

            if remaining_labels.shape[0] > 10:
                x_axis.append(alphas[e])
                metric_tmp = ece(remaining_logits, remaining_labels)[0].item()
                frac = remaining_labels.shape[0] / logits_ood_test.shape[0]
                metric_alpha.append(metric_tmp)
                fracs.append(frac)

    return x_axis, metric_alpha, fracs


def compute_metric(
    score_iid_val,
    score_iid_test,
    score_ood_test,
    data_iid_val,
    logits_iid_val,
    labels_iid_val,
    data_iid_test,
    logits_iid_test,
    labels_iid_test,
    data_ood_test,
    logits_ood_test,
    labels_ood_test,
    metrics=["intersection_fraction"],
    num_alphas=1000,
    ece_bins=10
):
    """
    Returns a dictionary of different queried metrics
    Arguments:
        - scores_iid_val: numpy array of scores on in distribution data (validation set)
        - scores_iid_test: numpy array of scores on in distribution data (test set)
        - scores_ood_test: numpy array of scores on ood data
        - data_iid_val: input data from training distribution (i.e. features of classifier)
        - logits_iid_val: logits of classifier on iid validation set 
        - labels_iid_val: ground truth labels on iid validation set 
        - data_iid_test: input data from iid test distribution (i.e. features of classifier)
        - logits_iid_test: logits of classifier on iid test set 
        - labels_iid_test: ground truth labels on iid iid set 
        - data_ood_test: input data from test distribution (i.e. features of classifier)
        - logits_ood_test: logits of classifier on data_test
        - labels_ood_test: ground truth labels for data_test
        - metrics: list of metrics that evaluate safe region 
        - num_alphas: amount of quantiles on which metrics are computed
        - ece_bins: number of bins to compute ece
    Return:
        - out: dictionary with (a) name of metric as key and (b) computed value of this metric
    """

    possible_metrics = [
        "intersection_fraction",
        "intersection_fraction_denoised",
        "intersection_alpha",
        "ausc_alpha",
        "dot_product",
        "dot_product_corrected",
        "dot_product_corrected_05",
        "ausc_alpha_normalized",
        "ausc_alpha_shifted",
        "ausc_fracs_shifted",
        "ausc_fracs",
        "ausc_fracs_normalized",
        "intersection_fraction_test",
        "acc_val",
        "acc_test",
        "ece_ood_test",
        "ece_iid_test",
        "ausc_ece_alpha",
        "ausc_ece_alpha_shifted"
    ]
    possible_metrics = [
        "accuracy",
        "ece",
        'ausc_alpha_ece',
        'ausc_alpha_acc',
        'ausc_fracs_ece',
        'ausc_fracs_acc',
        'fracs'
    ]
    assert all(metric in possible_metrics for metric in metrics)

    # Compute true-false values on ood test data
    true_false_ood_test = (
        (torch.from_numpy(logits_ood_test).argmax(1) == torch.from_numpy(labels_ood_test))
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )
    # Compute true-false values on iid val data
    true_false_iid_val = (
        (torch.from_numpy(logits_iid_val).argmax(1) == torch.from_numpy(labels_iid_val))
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )
    # Compute true-false values on iid test data
    true_false_iid_test = (
        (torch.from_numpy(logits_iid_test).argmax(1) == torch.from_numpy(labels_iid_test))
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )

    # Compute accuracy on training and test distribution
    accuracy_ood_test = np.sum(true_false_ood_test) / true_false_ood_test.shape[0]
    accuracy_iid_val = np.sum(true_false_iid_val) / true_false_iid_val.shape[0]
    accuracy_iid_test = np.sum(true_false_iid_test) / true_false_iid_test.shape[0]

    ece = _ECELoss(n_bins=ece_bins)
    metric_ece_ood_test = ece(logits_ood_test, labels_ood_test)[0].item()
    metric_ece_iid_val = ece(logits_iid_val, labels_iid_val)[0].item()
    metric_ece_iid_test = ece(logits_iid_test, labels_iid_test)[0].item()

    # Compute curves
    alphas_ood, accs_ood, fracs_ood = compute_curves(
        score_iid_val,
        score_ood_test,
        logits_ood_test,
        labels_ood_test,
        num_alphas=num_alphas,
        metric='acc'
    )
    
    alphas_iid, accs_iid, fracs_iid = compute_curves(
        score_iid_val,
        score_iid_test,
        logits_iid_test,
        labels_iid_test,
        num_alphas=num_alphas,
        metric='acc'
    )
    
    alphas_ece_ood, metrics_ece_ood, fracs_ece_ood = compute_curves(
        score_iid_val,
        score_ood_test,
        logits_ood_test,
        labels_ood_test,
        num_alphas=num_alphas,
        metric='ece',
        ece_bins=ece_bins
    )

    alphas_ece_iid, metrics_ece_iid, fracs_ece_iid = compute_curves(
        score_iid_val,
        score_iid_test,
        logits_iid_test,
        labels_iid_test,
        num_alphas=num_alphas,
        metric='ece',
        ece_bins=ece_bins
    )
    
    # Output dictionary
    out = {}

    if "accuracy" in metrics:
        out["acc_ood_test"] = accuracy_ood_test
        out["acc_iid_test"] = accuracy_iid_test
        out["acc_iid_val"] = accuracy_iid_val

    if "ece" in metrics:
        out['ece_ood_test'] = metric_ece_ood_test
        out['ece_iid_test'] = metric_ece_iid_test 
        out['ece_iid_val'] = metric_ece_iid_val

    #  Area Under Safe Curve for ece and quantiles alpha
    if 'ausc_alpha_ece' in metrics:
        out["ausc_alpha_ece_ood_test"] = auc(alphas_ece_ood, metrics_ece_ood)
        out["ausc_alpha_ece_iid_test"] = auc(alphas_ece_iid, metrics_ece_iid)

        # Shifted  Version
        out["ausc_alpha_ece_ood_test_shifted"] = out["ausc_alpha_ece_ood_test"] - (alphas_ece_ood[-1] - alphas_ece_ood[0]) * metric_ece_ood_test
        out["ausc_alpha_ece_iid_test_shifted"] = out["ausc_alpha_ece_iid_test"] - (alphas_ece_iid[-1] - alphas_ece_iid[0]) * metric_ece_iid_test

    #  Area Under Safe Curve for ece and fractions in S_alpha
    if 'ausc_fracs_ece' in metrics:

        out["ausc_fracs_ece_ood_test"] = auc(fracs_ece_ood, metrics_ece_ood)
        out["ausc_fracs_ece_iid_test"] = auc(fracs_ece_iid, metrics_ece_iid)
        
        # Shifted  Version
        out["ausc_fracs_ece_ood_test_shifted"] = out["ausc_fracs_ece_ood_test"] - (fracs_ece_ood[-1] - fracs_ece_ood[0]) * metric_ece_ood_test
        out["ausc_fracs_ece_iid_test_shifted"] = out["ausc_fracs_ece_iid_test"] - (fracs_ece_iid[-1] - fracs_ece_iid[0]) * metric_ece_iid_test

    #  Area Under Safe Curve for accuracy and quantiles alpha
    if 'ausc_alpha_acc' in metrics:

        out["ausc_alpha_ood_test"] = auc(alphas_ood, accs_ood)
        out["ausc_alpha_iid_test"] = auc(alphas_iid, accs_iid)

        # Shifted  Version
        out["ausc_alpha_ood_test_shifted"] = (
            out["ausc_alpha_ood_test"] - (alphas_ood[-1] - alphas_ood[0]) * accuracy_ood_test
        )
        out["ausc_alpha_iid_test_shifted"] = (
            out["ausc_alpha_iid_test"]- (alphas_iid[-1] - alphas_iid[0]) * accuracy_iid_test
        )
        
    #  Area Under Safe Curve for accuracy and fractions
    if "ausc_fracs_acc" in metrics:
        out["ausc_fracs_ood_test"] = auc(fracs_ood, accs_ood)
        out["ausc_fracs_iid_test"] = auc(fracs_iid, accs_iid)

        # Shifted  Version
        out["ausc_fracs_ood_test_shifted"] = (
            out["ausc_fracs_ood_test"] - (fracs_ood[-1] - fracs_ood[0]) * accuracy_ood_test
        )
        out["ausc_fracs_iid_test_shifted"] = (
            out["ausc_fracs_iid_test"] - (fracs_iid[-1] - fracs_iid[0]) * accuracy_iid_test
        )

    # Information about fractions
    if "fracs" in metrics:

        out["intersection_fraction_ood_test"] = first_below_line(
            fracs_ood, accs_ood, accuracy_iid_val, non_below_default="last_x"
        )
        out["intersection_alpha_ood_test"] = first_below_line(
            alphas_ood, accs_ood, accuracy_iid_val, non_below_default="last_x"
        )

        out["intersection_fraction_iid_test"] = first_below_line(
            fracs_iid, accs_iid, accuracy_iid_val, non_below_default="last_x"
        )
        out["intersection_alpha_iid_test"] = first_below_line(
            alphas_iid, accs_iid, accuracy_iid_val, non_below_default="last_x"
        )

        out['frac_remaining_ood_test'] = fracs_ood[-1]
        out['frac_remaining_iid_test'] = fracs_iid[-1]

    return out


def first_below_line(x_axis, y_values, line, non_below_default="last_x"):
    """
    Computes the x-value for which a Curve undercuts a line
    Arguments:
        - x_axis: x-values of curve
        - y_values: y-values of curve
        - line: line we consider (given by one value)
        - non_below_default: default value if line is not undercut
            if this value is 'last_x' it returns the last value of the x_axis
    """

    # if curve already starts below line we return 0
    if y_values[0] < line:
        return 0

    for e, x in enumerate(x_axis):
        if y_values[e] < line:
            return x

    if non_below_default == "last_x":
        return x_axis[-1]
    else:
        return non_below_default


@click.command()
# Required
@click.option(
    "--data_dir",
    help="Path to data directory",
    metavar="STRING",
    required=True,
)
@click.option(
    "--model_dir",
    help="Path to model directory",
    metavar="STRING",
    required=True,
)
@click.option(
    "--dataset",
    help="Dataset/DG task to evaluate the models on.",
    metavar="STRING",
    required=True,
)
@click.option(
    "--algorithm",
    help="Dataset/DG task to evaluate the models on.",
    metavar="STRING",
    required=True,
)
def main(**kwargs):
    opts = SimpleNamespace(**kwargs)

    tables = {}
    try:
        with open(opts.model_dir + "/metrics.pkl", "rb") as f:
            metrics_computed = pickle.load(f)
    except:
        metrics_computed = {}
        metrics_computed[opts.dataset] = {}

    metrics_computed[opts.dataset][opts.algorithm] = {}

    n_test_set = []
    for file_name in os.listdir(opts.model_dir):
        if (opts.algorithm in file_name) and (opts.dataset in file_name):
            n_test_set.append(
                int(re.search("domain(.*)_algorithm", file_name).group(1))
            )
    n_test_set = set(n_test_set)

    for test_domain in n_test_set:
        metrics_computed[opts.dataset][opts.algorithm][test_domain] = {}
        models = []
        for file_name in os.listdir(opts.model_dir):

            if (
                (opts.algorithm in file_name)
                and (opts.dataset in file_name)
                and (f"domain{test_domain}" in file_name)
            ):
                models.append(
                    load_model_from_path(file_name, opts.data_dir, opts.model_dir)
                )

        # Loading data
        (
            _,
            _,
            _,
            x_val,
            logits_val,
            y_val,
            x_test,
            logits_test,
            y_test,
        ) = load_data(opts.algorithm, opts.dataset, test_domain, opts.data_dir)

        model = model_selection(models, x_val, y_val)

        metrics = [
            "intersection_fraction_test",
            "intersection_fraction",
            "dot_product",
            "ausc_alpha_normalized",
            "ausc_alpha_shifted",
            "ausc_fracs",
            "ausc_fracs_shifted",
        ]
        outs = compute_metric(
            model,
            x_val,
            logits_val,
            y_val,
            x_test,
            logits_test,
            y_test,
            metrics=metrics,
        )

        metrics_computed[opts.dataset][opts.algorithm][test_domain] = dict(outs)

    with open(opts.model_dir + "/metrics.pkl", "wb") as f:
        pickle.dump(metrics_computed, f)


if __name__ == "__main__":
    ## TODO:  Have to  check  whether the main function still works (from old suprisaal-repository)
    main()
