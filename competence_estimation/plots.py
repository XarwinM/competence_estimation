"""
Module that offers different plot functions
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import _ECELoss



def plot(
    score_iid,
    score_ood,
    logits_val,
    labels_val,
    logits_test,
    labels_test,
    ax,
    metric='accuracy',
    color="green",
    fraction=False,
    label="",
    add_fraction_curve=False,
    comparison_lines=True,
    num_alphas=50
):
    """
    Plots G(alpha) / A(alpha) curve
    Arguments:
        - ....
    """
    
    assert metric in ['acc', 'ece']
    
        
    true_false_test = (
        (logits_test.cuda().argmax(1) == labels_test.cuda())
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )

    true_false_train = (
        (logits_val.cuda().argmax(1) == labels_val.cuda())
        .float()
        .cpu()
        .view(-1)
        .numpy()
    )
    
    alphas = np.linspace(0.0, 1, num_alphas)
    qs = np.quantile(score_iid, alphas)

    if metric == 'acc':
        metric_test = np.sum(true_false_test) / true_false_test.shape[0]
        metric_train = np.sum(true_false_train) / true_false_train.shape[0]
    elif metric == 'ece':
        ece = _ECELoss(n_bins=10)
        metric_test = ece(logits_test, labels_test)[0].item()
        metric_train = ece(logits_val, labels_val)[0].item()

    metric_alpha = []
    fracs = []
    x_axis = []
    for e, q in enumerate(qs):
        
        if metric == 'acc':
            remaining = true_false_test[score_ood <= q]


            if remaining.shape[0] > 10:
                x_axis.append(alphas[e])
                metric_tmp = np.sum(remaining) / remaining.shape[0]

                frac = remaining.shape[0] / true_false_test.shape[0]
                metric_alpha.append(metric_tmp)
                fracs.append(frac)
        if metric == 'ece':
            remaining_labels = labels_test[score_ood <= q]
            remaining_logits = logits_test[score_ood <= q]

            if remaining_labels.shape[0] > 10:
                x_axis.append(alphas[e])
                metric_tmp = ece(remaining_logits, remaining_labels)[0].item()
                frac = remaining_labels.shape[0] / logits_test.shape[0]
                metric_alpha.append(metric_tmp)
                fracs.append(frac)

    if fraction:
        ax.plot(fracs, metric_alpha, color=color, lw=3, alpha=0.3, label=label)
    else:
        ax.plot(x_axis, metric_alpha, color=color, lw=3, alpha=0.3, label=label)
        if add_fraction_curve:
            ax.plot(x_axis, fracs, color="blue", lw=3, label="Fraction Data", alpha=0.3)

    if comparison_lines:
        ax.axhline(metric_test, color="red", linestyle="dashed")
        ax.axhline(metric_train, color="black", linestyle="dashed")

    ax.grid(alpha=0.3)
    return ax
