"""
Module that offers different plot functions
"""

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

# Color scheme
colors = {'Deep-KNN': 'tab:orange', 'Softmax': '#50a050', 'ViM': '#5050a0', 'Logit': 'orange', 'GMM':  'green'}


# Env dictionary for data sets and different domains
env_dictionary = {'PACS': {0:'Art', 1:'Cartoon', 2:'Photo', 3: 'Sketch'},
    'VLCS':{0:'Cal101', 1:'LabelMe', 2:'SUN09', 3:'VOC2007'},
    'OfficeHome':{0:'Art',1:'Clipart', 2:'Product', 3:'Real World'},
    'TerraIncognita':{ 0:'L100', 1: 'L38', 2:'L43', 3: 'L46'},
    'DomainNet':{ 0:'Clipart', 1: 'Infograph', 2:'Painting', 3: 'Quickdraw', 4:'Real', 5:'Sketch'},
    'SVIRO':{ 0:'aclass', 1: 'escape', 2:'hilux', 3: 'i3', 4:'lexus', 
            5: 'tesla', 6:'tiguan', 7:'tucson', 8:'x5', 9:'zoe'},

}

def plot_accuracy_coverage(results_scores, results_true_false, score_fcts, classifiers, data_set='PACS', domains=[0,1,2,3], axs=None):
    """
    Plot the accuracy and coverage for different scores and domains
    Here we average over all classifiers (differently trained models)
    Arguments:
        - results_scores: Dictionary with scores for each score-function, data_set, classifier, domain (with id/ood split and percentages of open-world outliers)
        - results_true_false: Dictionary with true/false labels for each score-function, data_set, classifier, domain (with id/ood split and percentages of open-world outliers)
        - scores: List of score-functions to consider
        - classifiers: List of classifiers to consider
        - data_set: Name of the data set
        - domains: List of domains to consider
        - axs: Axes to plot on
    Returns:
        - axs: Axes on which we plotted
    """
    

    if axs is None:
        fig, axs = plt.subplots(1, len(domains), figsize=(15, 5))
    else:
        num_rows, num_cols = axs.shape
        assert num_rows == len(domains) or num_cols == len(domains), "Number of axes must match number of domains"
        
        
    # percentags of 0.0 means not open-world outliers
    # Here we only consider the scenario with no open-world outliers
    percentages = 0.0

    # fore each score-function, we create one line-plot
    quantile_means_x, quantile_means_y = {}, {}

    # For each domain, we create a subplot
    for i_domain, domain in enumerate(domains):
        
        # For each score-function, we create one line-plot
        for score in score_fcts:
                
            accuracies, coverages = [], []
            quantiles_coverages = []
            quantiles_coverages_x = [] # Quantile w.r.t. fraction of data in competence region
            acc_ids, acc_oods = [], []
            
            # Average over all classifiers
            for classifier in classifiers:
                    
                # Load scores and true/false labels
                scores_ood = np.array(results_scores[score][data_set][classifier][domain][percentages]['ood_test'])
                y_true_ood = np.array(results_true_false[score][data_set][classifier][domain][percentages]['ood_test'])
                scores_id = np.array(results_scores[score][data_set][classifier][domain][percentages]['id_test'])
                y_true_id = np.array(results_true_false[score][data_set][classifier][domain][percentages]['id_test'])

                # Compute accuracy
                acc_id = y_true_id.sum() / len(y_true_id)
                acc_ood = y_true_ood.sum() / len(y_true_ood)

                # Sort the scores
                index = np.argsort(scores_ood)
                scores_ood = scores_ood[index]
                y_true_ood = y_true_ood[index]

                # get the quantiles
                quantiles = list(np.quantile(scores_id, [0.95]))
                    
                quantile_coverage_classifier, quantile_coverage_classifier_x = [], []
                
                accuracies_tmp, coverages_tmp = [], []
                # Starting from 5 samples, compute the coverage and accuracy
                # Iterate over the scores in strides of 5
                for i in range(5, len(scores_ood), 5):
                        
                    # Compute coverage and accuracy for the current score
                    coverage = (scores_ood[i] >= scores_ood).sum() / len(scores_ood)
                    accuracy = y_true_ood[scores_ood[i] >= scores_ood].sum() / (scores_ood[i] >= scores_ood).sum()
                    accuracies_tmp.append(accuracy)
                    coverages_tmp.append(coverage)
                        
                    # save 95 % quantile when encountered
                    if len(quantiles) >0:
                        if scores_ood[i] > quantiles[0] and scores_ood[i-1]>= quantiles[0]:
                            quantile_coverage_classifier.append(accuracy)
                            quantile_coverage_classifier_x.append(coverage)
                            del quantiles[0]
                                
                # Save accuracies and quantiles for current classifier and score
                accuracies_tmp = np.array(accuracies_tmp)
                coverages_tmp = np.array(coverages_tmp)
                accuracies.append(accuracies_tmp)
                coverages.append(coverages_tmp)
                    
                # Save quantiles for current score and classifier
                quantiles_coverages_x.append(quantile_coverage_classifier_x)
                quantiles_coverages.append(quantile_coverage_classifier)
                    
                # Save accuracy and coverage for current classifier on ood an id data
                acc_ids.append(acc_id)
                acc_oods.append(acc_ood)

                
            # Compute average accuracy and coverage over different classifiers
            accuracies  = np.mean(np.array(accuracies), axis=0)
            coverages = np.mean(np.array(coverages),axis=0)

            # Compute mean quantiles over different classifiers
            quantiles_coverage = np.array(quantiles_coverages)
            quantiles_coverage_x = np.array(quantiles_coverages_x)
            mean_x = np.mean(quantiles_coverage_x, axis=0)
            mean_y = np.mean(quantiles_coverage, axis=0)

            quantile_means_x[score] = mean_x
            quantile_means_y[score] = mean_y
                
            # Line plot of coverage-accuracye curve for one score                
            sns.lineplot(x=coverages, y=accuracies, label=score, color=colors[score], ax=axs[i_domain], lw=2.2)

            axs[ i_domain].set_xlabel('Coverage', fontsize=15)

            if  i_domain  == 0:    
                axs[i_domain].set_ylabel(data_set + "\n Accuracy", fontsize=15)
            else: 
                axs[i_domain].set_ylabel("")
                
            # set grid
            axs[i_domain].grid(axis='both', alpha=0.5)
                
            # Remove legend
            axs[i_domain].get_legend().remove()

            axs[i_domain].set_title('Domain ' + str(i_domain+1), fontsize=18)
            
            #for score in score_fcts:
            axs[i_domain].scatter(quantile_means_x[score], quantile_means_y[score],  s=65, color=colors[score], marker='o', label=f"95%-Percentile ({score})")

            # Font size labels and ticks
            axs[i_domain].tick_params(axis='both', which='major', labelsize=12)
            axs[i_domain].tick_params(axis='both', which='minor', labelsize=12)
                
        # Plot accuracies for id and ood data for the mean classifier
        axs[i_domain].axhline(np.mean(acc_oods), linestyle='dashed', color='black', alpha=0.7, lw=1.4, label='Accuracy-OOD')
        axs[i_domain].axhline(np.mean(acc_ids), linestyle='dotted', color='black', alpha=0.7, lw=1.4, label='Accuracy-ID')
    
    plt.legend(fontsize=30, ncol=4)
    plt.legend(ncol=4, fontsize=16, loc='lower center', bbox_to_anchor=(-1.45,  -0.5), borderaxespad=0.)

    #fig.tight_layout()
    #fig.savefig('acc_cov_curve.pdf', dpi=300, bbox_inches='tight')
    return fig, axs

    
def ecdf(x):
    """
    Compute the empirical cumulative distribution function (ECDF) for a given array x
    Arguments:
        - x: array of values
    Returns:
        - sort_idx: sorted indices of x
        - xs: sorted values of x
        - ys: ECDF values of x
    """
    sort_idx = np.argsort(x)
    xs = x[sort_idx]
    ys = np.arange(1, len(xs)+1) / float(len(xs))
    return sort_idx, xs, ys
    
def plot_accuracy_curve(results_scores, results_true_false, score_functions, data_set, classifier, domain=0, axarr=None):
    """
    Plot the accuracy curve for a given data set, classifier and domain depending on the score function (and Competence Region defined via the score function)
    
    Arguments:
        - results_scores: dictionary containing the scores for each data set, classifier, domain and percentage of data
        - results_true_false: dictionary containing the true and false labels for each data set, classifier, domain and percentage of data
        - score_functions: list of score functions to consider
        - data_set: name of data set
        - classifier: name of classifier
        - domain: domain to consider (e.g. 0 or 1)
        - axarr: axis object to plot on
    Returns:
        - axarr: axis object containing the plot
    """
    
    # Color codes for Accuracy curve
    color_codes_scores = {
    'ood_test': '#800000',
    'id_test': '#000080',

    }

    # Color Codes for Accuracies of classifier
    color_codes_accs = {
        'ood_test': '#800000',
        'id_test': '#000080',
    }

    
    # Percentage of data to cut out / remove (due to noisiness when considering to few samples)
    cut_perc = 0.01
    
    if axarr is None:
        f, axarr = plt.subplots(1, len(score_functions), figsize=(15, 5))
    else:
        num_rows, num_cols = axarr.shape
        assert num_rows == len(score_functions) or num_cols == len(score_functions), "Number of axes must match number of score functions"
        
    # Loop through scores for given data set, classifier and domain
    for i, (score, ax) in enumerate(zip(score_functions, axarr.flat[:len(score_functions)])):
        
        # Loop through curves and standardize
        mean = results_scores[score][data_set][classifier][domain][0.0]['id_val'].mean()
        std = results_scores[score][data_set][classifier][domain][0.0]['id_val'].std()
        
        # Add 95% quantile
        score_val = (results_scores[score][data_set][classifier][domain][0.0]['id_val'] - mean) / std
        quantile = np.quantile(score_val, (0.05, 0.95))

        if i == 0:
            label_q = '95%-Percentile ' + r'(Validation)'
            label_acc = 'Accuracy curve' + r' ($A_{\alpha}$)'
            label_acc_id = 'Accuracy (ID)'
            label_acc_ood = 'Accuracy (OOD)'
        else:
            label_q = None
            label_acc = None
            label_acc_id = None
            label_acc_ood = None
        # Plot 95% quantile of id data (via validation set)
        ax.axvline(quantile[1], color='gray', linestyle='solid', lw=1.1, alpha=0.9, label=label_q)
        
        for code_s, code_a in zip(color_codes_scores.keys(), color_codes_accs.keys()):
            
            # Normalize scores 
            score_tests = results_scores[score][data_set][classifier][domain][0.0][code_s]
            score_tests_std = (score_tests - mean) / std
            
            # Sort values
            sort_idx, x, y = ecdf(score_tests_std)
            
            # Remove cases where percentage of data is too small
            show_idx = y > cut_perc
            
            if i == 0:
                label_frac = 'Coverage (OOD)' if 'ood' in code_s else 'Coverage (ID)'
            else:
                label_frac = None
            
            # Plot accuracies
            if code_a == 'ood_test':
                res = results_true_false[score][data_set][classifier][domain][0.0][code_a]
                cum_accuracy = np.cumsum(res[sort_idx]) / np.arange(1, len(res)+1)
                ax.plot(x[show_idx], cum_accuracy[show_idx], linestyle='solid', 
                        color=color_codes_accs[code_s], alpha=0.6, lw=3, label=label_acc)
        
                ax.plot(x[show_idx], y[show_idx], color=color_codes_scores[code_s], alpha=0.9, 
                        lw=2.8, linestyle='dotted', label=label_frac)
            # Plot ID accuracy
            else:
                ax.plot(x[show_idx], y[show_idx], color=color_codes_scores[code_s], alpha=0.3, 
                        lw=2.8, linestyle='dotted', label=label_frac)

        # Accuracy id
        acc_id = results_true_false[score][data_set][classifier][domain][0.0]['id_test'] 
        acc_id = np.sum(acc_id) / len(acc_id)
        ax.axhline(acc_id, linestyle='--', color='#000080', alpha=0.6, lw=1.4, label=label_acc_id)

        # Accuracy ood
        acc_od = results_true_false[score][data_set][classifier][domain][0.0]['ood_test']
        acc_od = np.sum(acc_od) / len(acc_od)
        ax.axhline(acc_od, linestyle='--', color='maroon', alpha=0.6, lw=1.4, label=label_acc_ood)
        
        # Prettify
        sns.despine(ax=ax)
        ax.set_title(score, fontsize=22)
        ax.grid(alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        if i == 0:
            ax.set_ylabel(f"{data_set} ({env_dictionary[data_set][domain]})", fontsize=18, labelpad=10)
        else:
            ax.set_ylabel('') 
            

    f.tight_layout()
    f.legend(ncol=3, fontsize=18, loc='lower center', bbox_to_anchor=(0.5, -0.15), borderaxespad=0.)
    return f, axarr