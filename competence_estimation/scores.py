"""
Module contains different score functions.
Most score functions are due https://github.com/haoqiwang/vim/blob/master/benchmark.py
"""

import numpy as np
import torch

import faiss
from scipy.special import softmax
from scipy.special import logsumexp

from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from scipy.special import logsumexp

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


def create_score_function(
    x_id_train,
    logits_id_train,
    y_id_train,
    x_id_val,
    logits_id_val,
    y_id_val,
    w,
    b,
    score_function="Softmax",
    **kwargs,
):
    """
    This function returns
        - the incompetence score on the validation set, and
        - a scoring function that computes the incompetence score for new features/logits.
    The sample size is denoted as N, the feature dimension as D, and the number of classes as L in the subsequent code.
    Arguments:
        - x_id_train: Input features of shape (N,D) from training set (in-distribution; id)
        - logits_id_train: Logits computed on features of shape (N,L) from training set (id)
        - y_id_train: Labels on training set of shape (N,) (iid)
        - x_id_val: Input features of shape (N,D) from validation set (id)
        - logits_id_val: Logits computed on features of shape (N,L) from validstion set (id)
        - y_id_val: Labels on validation set of shape (N,) (id)
        - w: weights of last layer
        - b: bias of last layer
        - score_function: Name of score function for which we want to return scores and scoring function
    Return:
        - scores_id: Scores on validation set
        - score_function: Score function that gets (features, logits) as input and returns the corresponding incompetence scores
    """

    # Supported OOD scores as incompetence scores
    assert score_function in [
        "Softmax",
        "Logit",
        "ViM",
        "Mahalanobis",
        "Deep-KNN",
        "Energy",
        "Energy-React",
        "GMM",
        "HBOS",
        "PCA",
    ]

    if score_function == "Softmax":
        score_iid = -softmax(logits_id_val, axis=-1).max(axis=-1)

        def score_function(features, logits):
            score = -softmax(logits, axis=-1).max(axis=-1)
            return score

    elif score_function == "PCA":
        from sklearn.decomposition import PCA

        assert kwargs["PCA"]["n_components"] < 1
        mod = PCA(n_components=kwargs["PCA"]["n_components"])
        mod.fit(x_id_train)
        rec_err = lambda x: np.mean(
            np.square((mod.inverse_transform(mod.transform(x)) - x)), axis=1
        )

        scores_id = rec_err(x_id_val)

        def score_function(feature, logits):
            scores = rec_err(feature)
            return scores

        return scores_id, score_function

    elif score_function == "GMM":
        import pyod.models.gmm as gmm

        clf = gmm.GMM(
            n_components=kwargs["GMM"]["n_components"],
            covariance_type="full",
            tol=0.001,
            reg_covar=1e-06,
            max_iter=100,
            n_init=1,
            init_params="kmeans",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            contamination=0.1,
        )
        clf.fit(x_id_train)
        score_iid = clf.decision_function(x_id_val)

        def score_function(feature, logit):
            score = clf.decision_function(feature)
            return score

    elif score_function == "HBOS":
        import pyod.models.hbos as hbos

        clf = hbos.HBOS(
            n_bins=kwargs["HBOS"]["n_bins"], alpha=0.1, tol=0.5, contamination=0.1
        )
        clf.fit(x_id_train)
        score_iid = clf.decision_function(x_id_val)

        def score_function(feature, logit):
            score = clf.decision_function(feature)
            return score

    elif score_function == "Energy-React":
        """
        score  due to  https://github.com/haoqiwang/vim/blob/master/benchmark.py
        """
        default_clip_quantile = 0.99
        clip = np.quantile(x_id_train, default_clip_quantile)

        logit_id_val_clip = np.clip(x_id_val, a_min=None, a_max=clip) @ w.T + b
        score_iid = -logsumexp(logit_id_val_clip, axis=-1)

        def score_function(features, logits):
            logit_clip = np.clip(features, a_min=None, a_max=clip) @ w.T + b
            score = -logsumexp(logit_clip, axis=-1)
            return score

    elif score_function == "Energy":
        """
        score  due to  https://github.com/haoqiwang/vim/blob/master/benchmark.py
        """
        score_iid = -logsumexp(logits_id_val, axis=-1)

        def score_function(features, logits):
            # for name, logit_ood in logit_oods.items():
            score = -logsumexp(logits, axis=-1)
            return score

    elif score_function == "Logit":
        score_iid = -logits_id_val.max(axis=-1)

        def score_function(features, logits):
            score = -logits.max(axis=-1)
            return score

    elif score_function == "Deep-KNN":
        """
        Deep-KNN scores due to https://github.com/deeplearning-wisc/knn-ood
        """

        K = kwargs["knn"]["K"]
        ftrain = normalizer(x_id_train)
        fval = normalizer(x_id_val)

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

    elif score_function == "Mahalanobis":
        """
        Mahalanobis-Scores (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
        """

        train_means = []
        train_feat_centered = []
        for i in range(logits_id_val.shape[1]):
            fs = x_id_train[y_id_train == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = torch.from_numpy(np.array(train_means)).cuda().float()
        prec = torch.from_numpy(ec.precision_).cuda().float()

        score_id = -np.array(
            [
                (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                for f in torch.from_numpy(x_id_val).cuda().float()
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

    elif score_function == "ViM":
        """
        Vim Score (due to https://github.com/haoqiwang/vim/blob/master/benchmark.py)
        minor changes w.r.t. DIM variable (adjusted for smaller input dimensions than 1000 dim)
        """

        u = -np.matmul(pinv(w), b)
        feature_id_train = x_id_train
        feature_id_val = x_id_val

        logit_id_train = logits_id_train
        logit_id_val = logits_id_val

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
