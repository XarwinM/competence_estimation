"""
This Module allows to run a sweep over different datasets with various domains, algorithms and percentages of open-world outliers.
Different scoring functions are supported.
"""

import argparse
import pickle
import time

import numpy as np
import torch

import yaml

from competence_estimation.utils import load_data, get_network_weights, mix_open, ENVS_DIC
from competence_estimation.scores import create_score_function
from competence_estimation.metrics import compute_metric

# Metrics that are computed
metrics = ["accuracy", "quantile_95"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run a sweep over different datasets and algorithms"
    )
    parser.add_argument(
        "--data_dir",
        help="directory of features, logits, labels and network weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where results are saved",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--datasets", help="List of data sets", nargs="+", type=str, default=["PACS"]
    )
    parser.add_argument(
        "--algorithms",
        help="List of algorithms/classifiers",
        nargs="+",
        type=str,
        default=["ERM"],
    )
    parser.add_argument(
        "--score_fct",
        help="List of score functions",
        nargs="+",
        type=str,
        default=["Deep-KNN"],
    )
    parser.add_argument(
        "--percentages",
        help="List of percentages of open world samples",
        nargs="+",
        type=str,
        default=[0],
    )
    parser.add_argument(
        "--add_to_existing",
        help="Add NOT to existing .pickle file if 0 and add to existing pickle file else",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--recompute",
        help="if 0: do not recompute scores if they are already compute; else: do recompute scores",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    # Load config file for score functions
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    if args.add_to_existing:
        with open(f"{args.output_dir}/results.pickle", "rb") as handle:
            results = pickle.load(handle)
        with open(f"{args.output_dir}/results_scores.pickle", "rb") as handle:
            results_scores = pickle.load(handle)
    else:
        # Results are stored in a dictionary of dictionaries of dictionaries of dictionaries;
        # the first key is the score function, the second the dataset, the third the algorithm, the fourth the domain
        # Reusults ares the metrics computed on the scores, e.g. the accuracy on th 95% quantile of the scores
        results = {}

        # Results_scores is a dictionary of dictionaries of dictionaries of dictionaries;
        # the first key is the score function, the second the dataset, the third the algorithm, the fourth the domain
        # With this dictionary we store the scores computed on features, logits and network-weights
        results_scores = {}

    # Percentages of open-world outliers
    percentages = [float(p) for p in args.percentages]
    assert all([p >= 0 and p <= 1 for p in percentages])

    # Implementation of for loop could be more efficient
    for score_function_name in args.score_fct:
        if score_function_name not in results:
            results[score_function_name] = {}
            results_scores[score_function_name] = {}

        # Measure time it takes to compute scores
        start_time = time.time()
        for dataset in args.datasets:
            if dataset not in results[score_function_name]:
                results[score_function_name][dataset] = {}
                results_scores[score_function_name][dataset] = {}

            for algorithm in args.algorithms:
                if algorithm not in results[score_function_name][dataset]:
                    results[score_function_name][dataset][algorithm] = {}
                    results_scores[score_function_name][dataset][algorithm] = {}

                for test_domain in range(ENVS_DIC[dataset]):
                    if (
                        test_domain
                        not in results[score_function_name][dataset][algorithm]
                        or args.recompute
                    ):
                        results[score_function_name][dataset][algorithm][
                            test_domain
                        ] = {}
                        results_scores[score_function_name][dataset][algorithm][
                            test_domain
                        ] = {}

                        iid_train, iid_val, iid_test, ood_test = load_data(
                            algorithm, dataset, test_domain, args.data_dir, fast=False
                        )
                        w, b = get_network_weights(
                            algorithm, dataset, test_domain, args.data_dir
                        )

                        for e, p in enumerate(percentages):
                            if (
                                p
                                not in results[score_function_name][dataset][algorithm][
                                    test_domain
                                ]
                            ):
                                results[score_function_name][dataset][algorithm][
                                    test_domain
                                ][p] = {}
                                results_scores[score_function_name][dataset][algorithm][
                                    test_domain
                                ][p] = {}

                            if p == 0.0:
                                features_out, logits_out, labels_out = (
                                    ood_test[0],
                                    ood_test[1],
                                    ood_test[2],
                                )

                            else:
                                # If open world samples are considered, we need to mix them with the closed world samples
                                features_open = np.load(
                                    args.data_dir
                                    + "/"
                                    + dataset
                                    + "/"
                                    + f"test_env_{test_domain}/{algorithm}_features_open_world.npy"
                                )
                                logits_open = np.load(
                                    args.data_dir
                                    + "/"
                                    + dataset
                                    + "/"
                                    + f"test_env_{test_domain}/{algorithm}_logits_open_world.npy"
                                )
                                features_out, logits_out, labels_out = mix_open(
                                    ood_test[0],
                                    ood_test[1],
                                    ood_test[2],
                                    features_open,
                                    logits_open,
                                    percentage=p,
                                )

                            if e == 0:
                                if iid_train[0].shape[0] > 50_000:
                                    # If more than 50_000 training data: shorten training data
                                    features_fit = iid_train[0]
                                    torch.manual_seed(0)
                                    idx = torch.randperm(features_fit.shape[0])[:50_000]
                                    (
                                        scores_iid_val,
                                        score_function,
                                    ) = create_score_function(
                                        iid_train[0][idx],
                                        iid_train[1][idx],
                                        iid_train[2][idx],
                                        iid_val[0],
                                        iid_val[1],
                                        iid_val[2],
                                        w,
                                        b,
                                        score_function=score_function_name,
                                        **config,
                                    )
                                else:
                                    (
                                        scores_iid_val,
                                        score_function,
                                    ) = create_score_function(
                                        iid_train[0],
                                        iid_train[1],
                                        iid_train[2],
                                        iid_val[0],
                                        iid_val[1],
                                        iid_val[2],
                                        w,
                                        b,
                                        score_function=score_function_name,
                                        **config,
                                    )
                                # Not necessary to compute for p > 0
                                scores_iid_test = score_function(
                                    iid_test[0], iid_test[1]
                                )

                            scores_ood_test = score_function(features_out, logits_out)
                            scores_iid_train = score_function(
                                iid_train[0], iid_train[1]
                            )

                            results_scores[score_function_name][dataset][algorithm][
                                test_domain
                            ][p]["scores_iid_train"] = scores_iid_train
                            results_scores[score_function_name][dataset][algorithm][
                                test_domain
                            ][p]["scores_iid_val"] = scores_iid_val
                            results_scores[score_function_name][dataset][algorithm][
                                test_domain
                            ][p]["scores_iid_test"] = scores_iid_test
                            results_scores[score_function_name][dataset][algorithm][
                                test_domain
                            ][p]["scores_ood_test"] = scores_ood_test

                            outs = compute_metric(
                                scores_iid_val,
                                scores_iid_test,
                                scores_ood_test,
                                iid_val[1],
                                iid_val[2],
                                iid_test[1],
                                iid_test[2],
                                logits_out=logits_out,
                                labels_out=labels_out,
                                metrics=metrics,
                            )

                            results[score_function_name][dataset][algorithm][
                                test_domain
                            ][p] = outs

        print(
            f"Score function {score_function_name} took around {time.time() - start_time} seconds to compute"
        )

        with open(f"{args.output_dir}/results.pickle", "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{args.output_dir}/results_scores.pickle", "wb") as handle:
            pickle.dump(results_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Results saved")
