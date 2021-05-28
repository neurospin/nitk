#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:19:18 2021

@author: ed203246
"""
import numpy as np
import time

def search_light(mask_arr, arr_train, arr_test, y_train, y_test, estimator, radius, verbose=1000):
    """ Search light.

    Parameters
    ----------
    mask_arr : array(x, y, z)
        Mask of voxels to consider.
    arr_train : array(n_samples, x, y, z)
        Training data .
    arr_test : array(n_samples, x, y, z)
        DESCRIPTION.
    y_train : array(n_samples)
        DESCRIPTION.
    y_test : array(n_samples)
        DESCRIPTION.
    estimator : sklearn Estimator
        DESCRIPTION.
    radius : float
        DESCRIPTION.
    verbose : int, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    results : dict
        Classifier: dict(auc=array(mask_arr.shape), bacc=array(mask_arr.shape)).

    """
    from sklearn import base
    from nitk.image import make_sphere
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    n_voxels = mask_arr.sum()
    if base.is_classifier(estimator):
        results = dict(auc=np.zeros(mask_arr.shape), bacc=np.zeros(mask_arr.shape))
    elif base.is_regressor(estimator):
        assert 0, "base.is_regressor(estimator)"
    else:
        assert 0, "is_regressor is_classifier are both False"

    mask_ijk = np.where(mask_arr)
    start = time.time()

    for cpt, (i, j, k) in enumerate(zip(*mask_ijk)):
        pass
        if verbose > 0 and cpt > 0 and cpt % verbose == 0:
                elapsed_time = time.time() - start
                prop_done = cpt / n_voxels
                total_time = elapsed_time / prop_done
                print("%.2f%% (Elapsed time:%.1fs/%.1fs)" % (prop_done * 100, elapsed_time, total_time))

        ball_arr = make_sphere(mask_arr.shape, (i, j, k), radius)
        estimator.fit(arr_train[:, ball_arr], y_train)
        if base.is_classifier(estimator):
            y_pred_test =  estimator.predict(arr_test[:, ball_arr])
            score_pred_test =  estimator.decision_function(arr_test[:, ball_arr])
            results['auc'][i, j, k] = roc_auc_score(y_test, score_pred_test)
            results['bacc'][i, j, k] = balanced_accuracy_score(y_test, y_pred_test)

        elif base.is_regressor(estimator):
            assert 0, "base.is_regressor(estimator)"

    return results