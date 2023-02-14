"""
Module contains different score functions. Most score functions are due https://github.com/haoqiwang/vim/blob/master/benchmark.py
"""

import numpy as np
import torch
from tqdm import tqdm

import faiss
from sklearn.metrics import pairwise_distances_argmin_min
import torch.nn.functional as F
from scipy.special import softmax
from scipy.special import logsumexp

from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from scipy.special import logsumexp

from competence_estimation.utils import _mmd

from torch import nn


# Due to from https://github.com/haoqiwang/vim/blob/master/benchmark.py
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


def kl(p, q):
    """KL - divergence (from https://github.com/haoqiwang/vim/blob/master/benchmark.py)"""
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def create_score_function(
    x_iid_train,
    logits_iid_train,
    y_iid_train,
    x_iid_val,
    logits_iid_val,
    y_iid_val,
    w,
    b,
    score_function="max_softmax",
    **kwargs
):
    """
    Return score on validation set and scoring function that computes the OOD score for a given feature and logit.
    In the following N is the sample size and D the feature dimension, as well as L the number of classes
    Arguments:
        - x_iid_train: Input features of shape (N,D) from training set (iid) 
        - logits_iid_train: Logits computed on features of shape (N,L) from training set (iid)
        - y_iid_train: Labels on training set of shape (N,) (iid)
        - x_iid_val: Input features of shape (N,D) from validation set  (iid)
        - logits_iid_val: Logits computed on features of shape (N,L) from validstion set (iid)
        - y_iid_val: Labels on validation set of shape (N,) (iid)
        - w: layer of last layer
        - b: bias of last layer
        - score_function: Name of score function for which we want to return scores and scoring function
    Return:
        - scores_iid: Scores on validation set
        - score_function: Score function that gets (features, logits) as input and returns the corresponding score
    """

    assert score_function in ["max_softmax", "max_logit", "vim", "mahalanobis", "knn", 'energy', 'energy_react', 'GMM', 'HBOS', 'PCA']

    if score_function == "max_softmax":
        score_iid = -softmax(logits_iid_val, axis=-1).max(axis=-1)

        def score_function(features, logits):
            score = -softmax(logits, axis=-1).max(axis=-1)
            return score

    elif score_function == "PCA":
        import pyod.models.pca as pca
        clf = pca.PCA(n_components=None, n_selected_components=None, contamination=0.1, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, weighted=True, standardization=True)
        clf.fit(x_iid_train)
        score_iid =  clf.decision_function(x_iid_val)
        def score_function(feature, logit):
            score =  clf.decision_function(feature)
            return score


    elif score_function == 'GMM':
        import pyod.models.gmm as gmm
        if 'n_components' in kwargs:
            n_components = kwargs['n_components']
        else:
            n_components =1
        clf = gmm.GMM(n_components=n_components, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, contamination=0.1)
        clf.fit(x_iid_train)
        score_iid =  clf.decision_function(x_iid_val)
        def score_function(feature, logit):
            score =  clf.decision_function(feature)
            return score

    elif score_function == "HBOS":
        import  pyod.models.hbos as hbos
        if 'n_bins' in kwargs:
            n_bins = kwargs['n_bins']
            print(f"n_bins {n_bins} chosen")
        else:
            n_bins =1
        clf = hbos.HBOS(n_bins=n_bins, alpha=0.1, tol=0.5, contamination=0.1)
        clf.fit(x_iid_train)
        score_iid =  clf.decision_function(x_iid_val)
        def score_function(feature, logit):
            score =  clf.decision_function(feature)
            return score


    elif score_function == 'energy_react':
        """
        score  due to  https://github.com/haoqiwang/vim/blob/master/benchmark.py
        """
        default_clip_quantile = 0.99
        clip = np.quantile(x_iid_train, default_clip_quantile)

        logit_id_val_clip = np.clip(x_iid_val, a_min=None, a_max=clip) @ w.T + b
        score_iid = - logsumexp(logit_id_val_clip, axis=-1)
        def score_function(features, logits):
            logit_clip = np.clip(features, a_min=None, a_max=clip) @ w.T + b
            score = - logsumexp(logit_clip, axis=-1)
            return score


    elif score_function == 'energy':
        """
           score  due to  https://github.com/haoqiwang/vim/blob/master/benchmark.py
        """
        score_iid = -logsumexp(logits_iid_val, axis=-1)
        def score_function(features, logits):
            #for name, logit_ood in logit_oods.items():
            score = - logsumexp(logits, axis=-1)
            return score

    elif score_function == "max_logit":
        score_iid = -logits_iid_val.max(axis=-1)

        def score_function(features, logits):
            score = -logits.max(axis=-1)
            return score

    elif score_function == "knn":
        """
        scores due to https://github.com/deeplearning-wisc/knn-ood
        """
        if 'K' in kwargs:
            K = kwargs['K']
            print('K set to ', K)
        else:
            K = 1
        ftrain = normalizer(x_iid_train)
        fval = normalizer(x_iid_val)
        #fest = normalizer(x_test)

        index = faiss.IndexFlatL2(ftrain.shape[1])
        index.add(ftrain)

        D, _ = index.search(fval, K)
        scores_iid = -D[:, -1]

        def score_function(features, logits):

            fest = normalizer(features)
            D, _ = index.search(fest, K)
            scores_ood_test = -D[:, -1]
            return -scores_ood_test

        return -scores_iid, score_function

    elif score_function == "mahalanobis":
        """
        Mahalanobis-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
        """
        result = []

        train_means = []
        train_feat_centered = []
        for i in range(logits_iid_val.shape[1]):
            fs = x_iid_train[y_iid_train == i]  # .numpy()
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = torch.from_numpy(np.array(train_means)).cuda().float()  # .numpy()
        prec = torch.from_numpy(ec.precision_).cuda().float()  # .numpy()
        #mean = np.array(train_means)#.float()
        #prec = ec.precision_#.float()  # .numpy()

        score_id = -np.array(
            [
                (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                for f in torch.from_numpy(x_iid_val).cuda().float()
            ]
        )

        def score_function(features, logits):
            score_ood = -np.array(
                [
                    (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                    for f in torch.from_numpy(features).cuda().float()
                ]
            )
            return -score_ood

        return -score_id, score_function

    elif score_function == "vim":
        """
        Vim Score (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
        minor changes w.r.t. DIM variable (adjusted for smaller input dimensions than 1000 dim)
        """

        u = -np.matmul(pinv(w), b)
        feature_id_train = x_iid_train  # .numpy()
        feature_id_val = x_iid_val  # .numpy()

        logit_id_train = logits_iid_train  # .numpy()
        logit_id_val = logits_iid_val  # .numpy()

        result = []
        if feature_id_val.shape[-1] >= 2048:
            DIM = 1000
        elif feature_id_val.shape[-1] <= 2048 and feature_id_val.shape[-1] >= 1000:
            DIM = 512
        else:
            DIM = 256

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)

        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        score_id = -vlogit_id_val + energy_id_val

        def score_function(features, logits):

            energy_ood = logsumexp(logits, axis=-1)
            vlogit_ood = norm(np.matmul(features - u, NS), axis=-1) * alpha
            score_ood = -vlogit_ood + energy_ood
            return -score_ood

        return -score_id, score_function

    return score_iid, score_function


def score(
    x_train,
    logits_train,
    y_train,
    x_val,
    logits_val,
    # x_test_iid,
    # logits_test_iid,
    x_test,
    logits_test,
    w,
    b,
    score_function="kl",
    **kwargs
):
    """
    Computes scores chosen score_function.
    Arguments:
        - x_train: features on training data
        - logits_train: logits on training data
        - y_train: labels on training data
        - x_val: features on validation set (iid)
        - logits_val: logits on validation set (iid)
        - x_test: feauters on test set (ood)
        - logits_test: logits on test set (ood)
        - score_function: score function
        - kwargs: dictionary for arguments to compute score
    Return:
        - scores_iid: scores for iid validation  dataa
        - scores_ood: scores for ood data
    """

    assert score_function in [
        "kl",
        "knn",
        "knn_standard",
        "mahalanobis",
        "vim",
        "residual",
        "max_logit",
        "max_softmax",
    ]

    if score_function == "max_logit":
        return score_max_logit(logits_val.numpy(), logits_test.numpy())
    if score_function == "max_softmax":
        return score_max_softmax(logits_val.numpy(), logits_test.numpy())

    if score_function == "vim":
        return score_vim(
            x_train, logits_train, x_val, logits_val, x_test, logits_test, w, b
        )

    ##@@###########

    if score_function == "kl":
        return score_kl(x_train, logits_train, x_val, logits_val, x_test, logits_test)

    if score_function == "knn":
        return score_knn(x_train, x_val, x_test, **kwargs)

    if score_function == "knn_standard":
        return score_knn_standard(x_train, x_val, x_test, **kwargs)

    if score_function == "mahalanobis":
        return score_mahalanobis(
            x_train, logits_train, y_train, x_val, logits_val, x_test, logits_test
        )

    if score_function == "vim":
        return score_vim(
            x_train, logits_train, x_val, logits_val, x_test, logits_test, w, b
        )
    if score_function == "residual":
        return score_residual(
            x_train, logits_train, x_val, logits_val, x_test, logits_test, w, b
        )

    if score_function == "logits":
        # Has to be checked/corrected
        scores_iid = (1 / logits_val).max(1)[0].numpy()
        scores_ood = (1 / logits_test).max(1)[0].numpy()
        return scores_iid, scores_ood

    if score_function == "mmd_feature":
        # Has to be checked/corrected
        scores_iid, scores_ood = [], []
        for i in range(logits_val.shape[0]):
            with torch.no_grad():
                scores_iid.append(
                    _mmd(x_val[:128].cuda(), x_val[i : i + 1].cuda()).item()
                )
        for i in range(logits_test.shape[0]):
            with torch.no_grad():
                scores_ood.append(
                    _mmd(x_val[:128].cuda(), x_test[i : i + 1].cuda()).item()
                )
        return np.array(scores_iid), np.array(scores_ood)

    if score_function == "mmd_logit":
        # Has to be checked/corrected
        scores_iid, scores_ood = [], []
        for i in range(logits_val.shape[0]):
            with torch.no_grad():
                scores_iid.append(
                    _mmd(logits_val[:128].cuda(), logits_val[i : i + 1].cuda()).item()
                )
        for i in range(logits_test.shape[0]):
            with torch.no_grad():
                scores_ood.append(
                    _mmd(logits_val[:128].cuda(), logits_test[i : i + 1].cuda()).item()
                )
        return np.array(scores_iid), np.array(scores_ood)

    if score_function == "meta_mean":
        # Hast to be implemented with score_functions in kwargs
        summaries = ["max_softmax", "knn"]

        scores_iid_mean, scores_ood_mean = 0, 0
        for score_function_tmp in summaries:
            scores_iid, scores_ood = score(
                x_train,
                logits_train,
                y_train,
                x_val,
                logits_val,
                x_test,
                logits_test,
                w,
                b,
                score_function=score_function_tmp,
            )
            x_min, x_max = scores_iid.min(), scores_iid.max()
            scores_iid = (scores_iid - x_min) / (x_max - x_min)
            scores_ood = (scores_ood - x_min) / (x_max - x_min)
            scores_iid_mean += scores_iid
            scores_ood_mean += scores_ood

        return scores_iid_mean, scores_ood_mean

    if score_function == "meta_min":
        # See Meta_mean
        summaries = [
            "max_softmax",
            "knn2",
            "knn5",
        ]  # , 'knn', 'mahalanobis', 'residual']

        scores_iid, scores_ood = score(
            x_train,
            logits_train,
            y_train,
            x_val,
            logits_val,
            x_test,
            logits_test,
            w,
            b,
            score_function=summaries[0],
        )
        x_min, x_max = scores_iid.min(), scores_iid.max()
        scores_iid = (scores_iid - x_min) / (x_max - x_min)
        scores_ood = (scores_ood - x_min) / (x_max - x_min)
        scores_iid_mean, scores_ood_mean = scores_iid, scores_ood
        for score_function_tmp in summaries:
            scores_iid, scores_ood = score(
                x_train,
                logits_train,
                y_train,
                x_val,
                logits_val,
                x_test,
                logits_test,
                w,
                b,
                score_function=score_function_tmp,
            )
            x_min, x_max = scores_iid.min(), scores_iid.max()
            scores_iid = (scores_iid - x_min) / (x_max - x_min)
            scores_ood = (scores_ood - x_min) / (x_max - x_min)
            # print(scores_iid)
            scores_iid_mean = np.minimum(scores_iid, scores_iid_mean)
            scores_ood_mean = np.minimum(scores_ood, scores_ood_mean)

        return scores_iid_mean, scores_ood_mean


def score_knn(x_train, x_val, x_test, K=50):
    """
    scores for  k-nearest neighbors (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
    """
    ftrain = normalizer(x_train)
    fval = normalizer(x_val)
    fest = normalizer(x_test)
    # ftrain = x_train
    # fval = x_val
    # fest = x_test

    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)

    D, _ = index.search(fval, K)
    scores_iid_test = -D[:, -1]

    D, _ = index.search(fest, K)
    scores_ood_test = -D[:, -1]
    return -scores_iid_test, -scores_ood_test


def score_knn_standard(x_train, x_val, x_test, K=50):
    """
    scores for  k-nearest neighbors
    """

    ftrain = x_train
    fval = x_val
    fest = x_test

    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)

    D, _ = index.search(fval, K)
    scores_iid_test = -D[:, -1]

    D, _ = index.search(fest, K)
    scores_ood_test = -D[:, -1]
    return -scores_iid_test, -scores_ood_test


def score_kl(x_train, logits_train, x_val, logits_val, x_test, logits_test):
    """
    KL-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
    """
    result = []

    softmax_id_train = softmax(logits_train, axis=-1)  # .numpy()
    softmax_id_val = softmax(logits_val, axis=-1)  # .numpy()
    softmax_ood = softmax(logits_test, axis=-1)  # .numpy()

    pred_labels_train = np.argmax(softmax_id_train, axis=-1)

    mean_softmax_train = [
        softmax_id_train[pred_labels_train == i].mean(axis=0)
        for i in range(logits_val.shape[1])
    ]

    ## Add on to kl score - check nesseary
    for e in range(logits_val.shape[1]):
        if np.isnan(mean_softmax_train[e]).sum() > 0:
            mean_softmax_train[e] = np.zeros_like(mean_softmax_train[e])

    score_id = -pairwise_distances_argmin_min(
        softmax_id_val, np.array(mean_softmax_train), metric=kl
    )[1]

    score_ood = -pairwise_distances_argmin_min(
        softmax_ood, np.array(mean_softmax_train), metric=kl
    )[1]

    return np.array(-score_id), np.array(-score_ood)


def score_mahalanobis(
    x_train, logits_train, y_train, x_val, logits_val, x_test, logits_test
):
    """
    Mahalanobis-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
    """
    result = []

    train_means = []
    train_feat_centered = []
    for i in range(logits_val.shape[1]):
        fs = x_train[y_train == i].numpy()
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    mean = torch.from_numpy(np.array(train_means)).cuda().float()  # .numpy()
    prec = torch.from_numpy(ec.precision_).cuda().float()  # .numpy()

    score_id = -np.array(
        [
            (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in x_val.cuda().float()
        ]
    )
    score_ood = -np.array(
        [
            (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in x_test.cuda().float()
        ]
    )

    return -score_id, -score_ood


def score_vim(x_train, logits_train, x_val, logits_val, x_test, logits_test, w, b):
    """
    Vim-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
    """

    u = -np.matmul(pinv(w), b)
    feature_id_train = x_train.numpy()
    feature_id_val = x_val.numpy()
    feature_ood = x_test.numpy()

    logit_id_train = logits_train.numpy()
    logit_id_val = logits_val.numpy()
    logit_ood = x_test.numpy()

    result = []
    # Has been chaanged (256 was 512)
    if feature_id_val.shape[-1] >= 2048:
        DIM = 1000
    elif feature_id_val.shape[-1] <= 2048 and feature_id_val.shape[-1] >= 1000:
        DIM = 512
    else:
        DIM = 256

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)

    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    return -score_id, -score_ood


def score_residual(x_train, logits_train, x_val, logits_val, x_test, logits_test, w, b):
    """
    Residual-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
    """
    u = -np.matmul(pinv(w), b)
    feature_id_train = x_train.numpy()
    feature_id_val = x_val.numpy()
    feature_ood = x_test.numpy()

    logit_id_train = logits_train.numpy()
    logit_id_val = logits_val.numpy()
    logit_ood = x_test.numpy()

    result = []
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 256

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)

    score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
    return -score_id, -score_ood


def score_max_logit(logits_val, logits_ood):
    """
    Max-Logits-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
    """
    score_id = logits_val.max(axis=-1)
    score_ood = logits_ood.max(axis=-1)
    return -score_id, -score_ood


def score_max_softmax(logits_val, logits_ood):
    """
    Max-Softmax Score
    """
    score_id = softmax(logits_val, axis=-1).max(axis=-1)
    score_ood = softmax(logits_ood, axis=-1).max(axis=-1)
    return -score_id, -score_ood
