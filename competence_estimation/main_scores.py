import argparse
import pickle

import numpy as np

from competence_estimation.utils import load_data, get_network_weights, mix_open
from competence_estimation.scores  import create_score_function 
from competence_estimation.metrics  import compute_metric

# Number of Environments  for  different Datasets
ENVS_DIC = {
    "OfficeHome": 4,
    "VLCS": 4,
    "PACS": 4,
    "TerraIncognita": 4,
    "SVIRO": 10,
    "DomainNet": 6,
}

metrics  = [
    "accuracy",
    "ece",
    'ausc_alpha_ece',
    'ausc_alpha_acc',
    'ausc_fracs_ece',
    'ausc_fracs_acc',
    'fracs'
]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument("--datasets", nargs="+", type=str, default=["PACS"])
    parser.add_argument("--algorithms", nargs="+", type=str, default=["ERM"])
    parser.add_argument("--score_fct", nargs="+", type=str, default=["knn"])
    parser.add_argument("--percentages", nargs="+", type=str, default=[0])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--add_to_existing", type=int, default=1)
    args = parser.parse_args()


    if args.add_to_existing:
        print('Existing results loaded')
        with open(f"{args.output_dir}/results.pickle", 'rb') as handle:
            results = pickle.load(handle)
        with open(f"{args.output_dir}/results_scores.pickle", 'rb') as handle:
            results_scores = pickle.load(handle)
    else:
        results = {}
        results_scores = {}

    args.percentages = [float(p) for p in args.percentages]


    for score_function_name in args.score_fct:
        results[score_function_name] = {}
        results_scores[score_function_name] = {}

        for dataset in args.datasets:
            print('Dataset ', dataset)
            results[score_function_name][dataset] = {}
            results_scores[score_function_name][dataset] = {}

            for algorithm in args.algorithms:
                results[score_function_name][dataset][algorithm] = {}
                results_scores[score_function_name][dataset][algorithm] = {}

                for test_domain in range(ENVS_DIC[dataset]):
                    results[score_function_name][dataset][algorithm][test_domain] = {}
                    results_scores[score_function_name][dataset][algorithm][
                        test_domain
                    ] = {}

                    iid_train, iid_val, iid_test, ood_test = load_data(
                        algorithm, dataset, test_domain, args.data_dir, fast=False
                    )
                    w, b = get_network_weights(algorithm, dataset, test_domain, args.data_dir)

                    for e, p in enumerate(args.percentages):
                        results[score_function_name][dataset][algorithm][test_domain][
                            p
                        ] = {}
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

                        scores_iid_val, score_function = create_score_function(
                            iid_train[0],
                            iid_train[1],
                            iid_train[2],
                            iid_val[0],
                            iid_val[1],
                            iid_val[2],
                            w,
                            b,
                            score_function =  score_function_name,
                        )

                        scores_ood_test = score_function(features_out, logits_out)
                        if e == 0:
                            scores_iid_test = score_function(
                                iid_test[0], iid_test[1] 
                            )

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
                            iid_val[0],
                            iid_val[1],
                            iid_val[2],
                            iid_test[0],
                            iid_test[1],
                            iid_test[2],
                            features_out,
                            logits_out,
                            labels_out,
                            metrics=metrics,
                        )

                        results[score_function_name][dataset][algorithm][test_domain][p] = outs

    with open(f"{args.output_dir}/results.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{args.output_dir}/results_scores.pickle", 'wb') as handle:
        pickle.dump(results_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Results saved")
