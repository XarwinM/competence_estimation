"""
Module to create transformed scores from already computed scores
"""
import argparse
import pickle
import time

import yaml

from competence_estimation.classifiers import transform_scores
from competence_estimation.utils import load_data, ENVS_DIC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sweep")
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
        help="List of Classifiers/Algorithms",
        nargs="+",
        type=str,
        default=["ERM"],
    )
    parser.add_argument(
        "--score_fct",
        help="List of Score Functions",
        nargs="+",
        type=str,
        default=["knn"],
    )
    parser.add_argument(
        "--classifier_type",
        help="transformation of score: monoton or unrestricted",
        nargs="+",
        type=str,
        default=["monoton"],
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

    # Load already computed scores
    with open(f"{args.output_dir}/results_scores.pickle", "rb") as handle:
        results_scores = pickle.load(handle)

    if args.add_to_existing:
        with open(
            f"{args.output_dir}/results_scores_transformed.pickle", "rb"
        ) as handle:
            results_scores_transformed = pickle.load(handle)
    else:
        results_class_transformed = {}

    if args.recompute:
        recompute = True
    else:
        recompute = False

    percentages = [float(p) for p in args.percentages]
    assert all([p >= 0 and p <= 1 for p in percentages])

    # Run sweep over all score functions, datasets, algorithms, domains and percentages
    for score_function_name in args.score_fct:
        if score_function_name not in results_class_transformed:
            results_class_transformed[score_function_name] = {}

        # Measure time it takes to compute scores
        start_time = time.time()
        for dataset in args.datasets:
            if dataset not in results_class_transformed[score_function_name]:
                results_class_transformed[score_function_name][dataset] = {}

            for algorithm in args.algorithms:
                if (
                    algorithm
                    not in results_class_transformed[score_function_name][dataset]
                ):
                    results_class_transformed[score_function_name][dataset][
                        algorithm
                    ] = {}

                for classifier_type in args.classifier_type:
                    if (
                        classifier_type
                        not in results_class_transformed[score_function_name][dataset][
                            algorithm
                        ]
                    ):
                        results_class_transformed[score_function_name][dataset][
                            algorithm
                        ][classifier_type] = {}

                    for test_domain in range(ENVS_DIC[dataset]):
                        if (
                            test_domain
                            not in results_class_transformed[score_function_name][
                                dataset
                            ][algorithm][classifier_type]
                        ):
                            results_class_transformed[score_function_name][dataset][
                                algorithm
                            ][classifier_type][test_domain] = {}

                        first_compute = 0
                        for ep, p in enumerate(percentages):
                            if (
                                p
                                not in results_class_transformed[score_function_name][
                                    dataset
                                ][algorithm][classifier_type][test_domain]
                                or recompute
                            ):
                                results_class_transformed[score_function_name][dataset][
                                    algorithm
                                ][classifier_type][test_domain][p] = {}

                                # If scores are computed
                                scores_iid_val = results_scores[score_function_name][
                                    dataset
                                ][algorithm][test_domain][p]["scores_iid_val"]
                                scores_iid_test = results_scores[score_function_name][
                                    dataset
                                ][algorithm][test_domain][p]["scores_iid_test"]
                                scores_ood_test = results_scores[score_function_name][
                                    dataset
                                ][algorithm][test_domain][p]["scores_ood_test"]

                                if first_compute == 0:
                                    _, iid_val, iid_test, ood_test = load_data(
                                        algorithm,
                                        dataset,
                                        test_domain,
                                        args.data_dir,
                                        fast=False,
                                    )
                                    true_false_train = (
                                        iid_val[1].argmax(1) == iid_val[2]
                                    )

                                    (
                                        scores_iid_val_class,
                                        score_function_class,
                                    ) = transform_scores(
                                        scores_iid_val,
                                        true_false_train,
                                        classifier_type=classifier_type,
                                    )
                                    first_compute += 1

                                scores_iid_test_class = score_function_class(
                                    scores_iid_test
                                )
                                scores_ood_test_class = score_function_class(
                                    scores_ood_test
                                )
                                results_class_transformed[score_function_name][dataset][
                                    algorithm
                                ][classifier_type][test_domain][p][
                                    "scores_iid_val"
                                ] = scores_iid_val_class
                                results_class_transformed[score_function_name][dataset][
                                    algorithm
                                ][classifier_type][test_domain][p][
                                    "scores_iid_test"
                                ] = scores_iid_test_class
                                results_class_transformed[score_function_name][dataset][
                                    algorithm
                                ][classifier_type][test_domain][p][
                                    "scores_ood_test"
                                ] = scores_ood_test_class

        print(
            f"Score function {score_function_name} took around {time.time() - start_time} seconds to compute"
        )

        with open(
            f"{args.output_dir}/results_scores_transformed.pickle", "wb"
        ) as handle:
            pickle.dump(
                results_class_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        print("Results saved")
