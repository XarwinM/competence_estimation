import torch
import pyod

def score_func_names(ind=0):
    names = [
        "ABOD",  # geht langsam
        # "ALAD", tensorflow
        # "AnoGAN", tensorflow
        "AutoEncoder",
        #"CBLOF",  # geht langsam, clustern kann fehlschlagen
        "COF",
        # "CD", conditional
        "COPOD",  # slow
        # "DeepSVDD", tensorflow
        "ECOD",
        # "FeatureBagging", combo missing
        "GMM",
        "HBOS",
        "IForest",
        "KDE",
        "KNN",
        "KPCA",
        #"LMDD", slow
        "LODA",
        "LOF",
        #"LOCI", slow
        "LUNAR",  # slow
        # "LSCP", detector list?
        # "MAD", univariate
        "MCD",
        # "MO_GAAL", tensorflow
        "OCSVM",
        "PCA",
        # "RGraph", slow
        "Sampling",
        "SOD",
        # "SO_GAAL", tensorflow
        "SOS",
        # "SUOD", ModuleNotFoundError: No module named 'suod'
        # "VAE", tensorflow
        #"XGBOD" xgboost
    ]
    return names[ind]

def score(
    x_train,
    logits_train = None,
    y_train = None,
    x_val = None,
    logits_val = None,
    x_test_iid = None,
    logits_test_iid = None,
    x_test = None,
    logits_test = None,
    w = None,
    b = None,
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
        "ABOD",
        "ALAD",
        "AnoGAN",
        "AutoEncoder",
        "CBLOF",
        "COF",
        "CD",
        "COPOD",
        "DeepSVDD",
        "ECOD",
        "FeatureBagging",
        "GMM",
        "HBOS",
        "IForest",
        "KDE",
        "KNN",
        "KPCA",
        "LMDD",
        "LODA",
        "LOF",
        "LOCI",
        "LUNAR",
        "LSCP",
        "MAD",
        "MCD",
        "MO_GAAL",
        "OCSVM",
        "PCA",
        "RGraph",
        "Sampling",
        "SOD",
        "SO_GAAL",
        "SOS",
        "SUOD",
        "VAE",
        "XGBOD"
    ]


    if score_function == "ABOD":
        import pyod.models.abod as abod
        clf = abod.ABOD(contamination=0.1, n_neighbors=5, method='fast')

    if score_function == "ALAD":
        import pyod.models.alad as alad
        clf = alad.ALAD(activation_hidden_gen='tanh', activation_hidden_disc='tanh', output_activation=None, dropout_rate=0.2, latent_dim=2, dec_layers=[5, 10, 25], enc_layers=[25, 10, 5], disc_xx_layers=[25, 10, 5], disc_zz_layers=[25, 10, 5], disc_xz_layers=[25, 10, 5], learning_rate_gen=0.0001, learning_rate_disc=0.0001, add_recon_loss=False, lambda_recon_loss=0.1, epochs=200, verbose=0, preprocessing=False, add_disc_zz_loss=True, spectral_normalization=False, batch_size=32, contamination=0.1)

    if score_function == "AnoGAN":
        import pyod.models.anogan as anogan
        clf = anogan.AnoGAN(activation_hidden='tanh', dropout_rate=0.2, latent_dim_G=2, G_layers=[20, 10, 3, 10, 20], verbose=0, D_layers=[20, 10, 5], index_D_layer_for_recon_error=1, epochs=500, preprocessing=False, learning_rate=0.001, learning_rate_query=0.01, epochs_query=20, batch_size=32, output_activation=None, contamination=0.1)

    if score_function == "AutoEncoder":
        import pyod.models.auto_encoder_torch as auto_encoder
        clf = auto_encoder.AutoEncoder(hidden_neurons=None, hidden_activation='relu', batch_norm=True, learning_rate=0.001, epochs=100, batch_size=32, dropout_rate=0.2, weight_decay=1e-05, preprocessing=True, loss_fn=None, contamination=0.1, device=None)

    if score_function == "CBLOF":
        import pyod.models.cblof as cblof
        clf = cblof.CBLOF(n_clusters=8, contamination=0.1, clustering_estimator=None, alpha=0.9, beta=5, use_weights=False, check_estimator=False, random_state=None, n_jobs=1)

    if score_function == "COF":
        import pyod.models.cof as cof
        clf = cof.COF(contamination=0.1, n_neighbors=20, method='fast')

    if score_function == "CD":
        import pyod.models.cd as cd
        clf = cd.CD(whitening=True, contamination=0.1, rule_of_thumb=False)

    if score_function == "COPOD":
        import pyod.models.copod as copod
        clf = copod.COPOD(contamination=0.1, n_jobs=1)

    if score_function == "DeepSVDD":
        import pyod.models.deep_svdd as deep_svdd
        clf = deep_svdd.DeepSVDD(c=None, use_ae=False, hidden_neurons=None, hidden_activation='relu', output_activation='sigmoid', optimizer='adam', epochs=100, batch_size=32, dropout_rate=0.2, l2_regularizer=0.1, validation_size=0.1, preprocessing=True, verbose=1, random_state=None, contamination=0.1)

    if score_function == "ECOD":
        import  pyod.models.ecod as ecod
        clf = ecod.ECOD(contamination=0.1, n_jobs=1)

    if score_function == "FeatureBagging":
        import pyod.models.feature_bagging as FeatureBagging
        clf = feature_bagging.FeatureBagging(base_estimator=None, n_estimators=10, contamination=0.1, max_features=1.0, bootstrap_features=False, check_detector=True, check_estimator=False, n_jobs=1, random_state=None, combination='average', verbose=0, estimator_params=None)

    if score_function == "GMM":
        import pyod.models.gmm as gmm
        #clf = gmm.GMM(n_components=kwargs['n_components'], covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, contamination=0.1)
        clf = gmm.GMM(n_components=kwargs['n_components'], covariance_type=kwargs['covariance_type'], tol=0.001, reg_covar=1e-06, max_iter=kwargs['iterations'], n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, contamination=0.1)

    if score_function == "HBOS":
        import  pyod.models.hbos as hbos
        clf = hbos.HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)

    if score_function == "IForest":
        import pyod.models.iforest as iforest
        clf = iforest.IForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1, behaviour='old', random_state=None, verbose=0)

    if score_function == "KDE":
        import  pyod.models.kde as kde
        clf = kde.KDE(contamination=0.1, bandwidth=1.0, algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None)

    if score_function == "KNN":
        import  pyod.models.knn as knn
        clf = knn.KNN(contamination=0.1, n_neighbors=5, method='largest', radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1, **kwargs)

    if score_function == "KPCA":
        import  pyod.models.kpca as kpca
        clf = kpca.KPCA(contamination=0.1, n_components=None, n_selected_components=None, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False, copy_X=True, n_jobs=None, sampling=False, subset_size=20, random_state=None)

    if score_function == "LMDD":
        import  pyod.models.lmdd as lmdd
        clf = lmdd.LMDD(contamination=0.1, n_iter=50, dis_measure='aad', random_state=None)

    if score_function == "LODA":
        import pyod.models.loda as loda
        clf = loda.LODA(contamination=0.1, n_bins=10, n_random_cuts=100)

    if score_function == "LOF":
        import pyod.models.lof as lof
        clf = lof.LOF(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=0.1, n_jobs=1, novelty=True)#

    if score_function == "LOCI":
        import pyod.models.loci as loci
        clf = loci.LOCI(contamination=0.1, alpha=0.5, k=3)

    if score_function == "LUNAR":
        import pyod.models.lunar as lunar
        clf = lunar.LUNAR(model_type='WEIGHT', n_neighbours=5, negative_sampling='MIXED', val_size=0.1, epsilon=0.1, proportion=1.0, n_epochs=200, lr=0.001, wd=0.1, verbose=0) # scaler=MinMaxScaler()

    if score_function == "LSCP":
        import pyod.models.lscp as lscp
        clf = lscp.LSCP(detector_list, local_region_size=30, local_max_features=1.0, n_bins=10, random_state=None, contamination=0.1)

    if score_function == "MAD":
        import pyod.models.mad as mad
        clf = mad.MAD(threshold=3.5)

    if score_function == "MCD":
        import pyod.models.mcd as mcd
        clf = mcd.MCD(contamination=0.1, store_precision=True, assume_centered=False, support_fraction=None, random_state=None)

    if score_function == "MO_GAAL":
        import pyod.models.mo_gaal as mo_gaal
        clf = mo_gaal.MO_GAAL(k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, momentum=0.9, contamination=0.1)

    if score_function == "OCSVM":
        import pyod.models.ocsvm as ocsvm
        clf = ocsvm.OCSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, contamination=0.1)

    if score_function == "PCA":
        import pyod.models.pca as pca
        clf = pca.PCA(n_components=None, n_selected_components=None, contamination=0.1, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, weighted=True, standardization=True)

    if score_function == "RGraph":
        import pyod.models.rgraph as rgraph
        clf = rgraph.RGraph(transition_steps=10, n_nonzero=10, gamma=50.0, gamma_nz=True, algorithm='lasso_lars', tau=1.0, maxiter_lasso=1000, preprocessing=True, contamination=0.1, blocksize_test_data=10, support_init='L2', maxiter=40, support_size=100, active_support=True, fit_intercept_LR=False, verbose=True)

    if score_function == "Sampling":
        import pyod.models.sampling as sampling
        clf = sampling.Sampling(contamination=0.1, subset_size=20, metric='minkowski', metric_params=None, random_state=None)

    if score_function == "SOD":
        import pyod.models.sod as sod
        clf = sod.SOD(contamination=0.1, n_neighbors=20, ref_set=10, alpha=0.8)

    if score_function == "SO_GAAL":
        import pyod.models.so_gaal as so_gaal
        clf = so_gaal.SO_GAAL(stop_epochs=20, lr_d=0.01, lr_g=0.0001, momentum=0.9, contamination=0.1)

    if score_function == "SOS":
        import pyod.models.sos as sos
        clf = sos.SOS(contamination=0.1, perplexity=4.5, metric='euclidean', eps=1e-05)

    if score_function == "SUOD":
        import pyod.models.suod as suod
        clf = suod.SUOD(base_estimators=None, contamination=0.1, combination='average', n_jobs=None, rp_clf_list=None, rp_ng_clf_list=None, rp_flag_global=True, target_dim_frac=0.5, jl_method='basic', bps_flag=True, approx_clf_list=None, approx_ng_clf_list=None, approx_flag_global=True, approx_clf=None, cost_forecast_loc_fit=None, cost_forecast_loc_pred=None, verbose=False)

    if score_function == "VAE":
        import pyod.models.vae as vae
        clf = vae.VAE(encoder_neurons=None, decoder_neurons=None, latent_dim=2, hidden_activation='relu', output_activation='sigmoid',  optimizer='adam', epochs=100, batch_size=32, dropout_rate=0.2, l2_regularizer=0.1, validation_size=0.1, preprocessing=True, verbose=1, random_state=None, contamination=0.1, gamma=1.0, capacity=0.0) #loss=<function mean_squared_error>,

    if score_function == "XGBOD":
        import pyod.models.xgbod as xgbod
        clf = xgbod.XGBOD(estimator_list=None, standardization_flag_list=None, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, **kwargs)

    clf.fit(x_train)
    return clf
    '''
    val_iid_scores = clf.decision_function(x_val)
    test_iid_scores = clf.decision_function(x_test_iid)

    ood_scores = clf.decision_function(x_test)
    return val_iid_scores, test_iid_scores,  ood_scores
    '''
    
