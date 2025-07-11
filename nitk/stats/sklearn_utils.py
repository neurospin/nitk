#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 01:33:09 2020

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import time
import sklearn.metrics as metrics



def fit_predict(estimators, X_train, y_train, X_test):
    """ Fit predict wrapper: gather prediction and other information 
    """
    pred_dict = {name:dict(estimator_type=est._estimator_type) for name, est in estimators.items()}
    for name in estimators:
        t0 = time.process_time()
        estimators[name].fit(X_train, y_train)
        pred_dict[name]["time"] = time.process_time() - t0
        pred_dict[name]["y_test"] = estimators[name].predict(X_test)
        if(len(np.unique(y)) == 2): # TODO extends for mulinomial classif
            if hasattr(estimators[name], 'decision_function'):
                pred_dict[name]["score_test"] = estimators[name].decision_function(X_test)
            elif hasattr(estimators[name], 'predict_log_proba'):
                pred_dict[name]["score_test"] = estimators[name].predict_log_proba(X_test)[:, 1]
        try:
            pred_dict[name]["coefs"] = estimators[name].coef_
        except:
            pass

        if hasattr(estimators[name], 'alpha_'):
            pred_dict[name]["best_param"] = estimators[name].alpha_
        elif hasattr(estimators[name], 'C_'):
            pred_dict[name]["best_param"] = estimators[name].C_[0]
        else:
            pred_dict[name]["best_param"] = np.NaN

    return pred_dict

    
def aggregate_predictions(cv_ret, cv, X, y):
    """ Aggregate by estimator, ie, convert list (CV) of dict (estimators)
        to dict(estimators) of predictions
    """
    assert cv.n_splits == len(cv_ret)
    estimator_names = set(list(np.array([[k for k in fold_dict.keys()] for fold_dict in cv_ret]).ravel()))
    preds = {name:dict(y_test=np.zeros(len(y)),
                       score_test=np.zeros(len(y)),
                       time=[], coefs=[], best_param=[]) for name in estimator_names}

    for i, (train, test) in enumerate(cv.split(X, y)):
        for name in estimator_names:
            preds[name]['y_test'][test] = cv_ret[i][name]['y_test']
            try:
                preds[name]['score_test'][test] = cv_ret[i][name]['score_test']
            except:
                if 'score_test' in preds[name]: preds[name].pop('score_test')
            try:
                preds[name]['time'].append(cv_ret[i][name]['time'])
            except:
                preds[name]['time'].append(None)
            try:
                preds[name]['best_param'].append(cv_ret[i][name]['best_param'])
            except:
                preds[name]['best_param'].append(None)                
            try:
                preds[name]["coefs"].append(cv_ret[i][name]['coefs'].ravel())
            except:
                if 'coefs' in preds[name]: preds[name].pop('coefs')
    return preds

def cv_summary_classif(cv_ret, cv, X, y):
    preds = _aggregate_predictions(cv_ret, cv, X, y)
    # Compute statistics
    #stats = {name:dict() for name in estimators}
    stats_list = list()
    stats_folds_list = list()

    for name in preds:
        accs_test = np.asarray([metrics.accuracy_score(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
        baccs_test = np.asarray([metrics.recall_score(y[test], preds[name]['y_test'][test], average=None).mean() for train, test in cv.split(X, y)])
        recalls_test = np.asarray([metrics.recall_score(y[test], preds[name]['y_test'][test], average=None) for train, test in cv.split(X, y)])
        aucs_test = np.asarray([metrics.roc_auc_score(y[test], preds[name]['score_test'][test]) for train, test in cv.split(X, y)])
        size_test = np.asarray([len(y[test]) for train, test in cv.split(X, y)])

        folds = pd.DataFrame()
        folds['model'] = [name] * len(baccs_test)
        folds['fold'] = ["CV%i" % fold for fold in range(len(baccs_test))]
        folds['bacc_test'] = baccs_test
        folds['auc_test'] = aucs_test
        for i, lev in enumerate(np.unique(y)):
            folds['recall%i_test' % lev] = recalls_test[:, i]
        folds["time"] = preds[name]['time']
        folds["best_param"] = preds[name]['best_param']
        folds["size_test"] = size_test
        stats_folds_list.append(folds)
        stats_list.append([name, np.mean(baccs_test), np.mean(aucs_test)] + \
                              np.asarray(recalls_test).mean(axis=0).tolist() + [np.mean(preds[name]['time']), str(preds[name]['best_param'])])
    stats_folds_df = pd.concat(stats_folds_list)
    stats_df = pd.DataFrame(stats_list, columns=['model', 'bacc_test', 'auc_test'] + \
                                                ['recall%i_test' % lev for lev in np.unique(y)] + ["time", "best_param"])
    stats_df["size"] = str(["%i:%i" % (lab, np.sum(y==lab)) for lab in np.unique(y)])

    model_params = None

    return stats_df, stats_folds_df, model_params

def cv_summary_regression(cv_ret, cv, X, y):
    preds = _aggregate_predictions(cv_ret, cv, X, y)
    # Compute statistics
    #stats = {name:dict() for name in estimators}
    stats_list = list()
    stats_folds_list = list()
    
    for name in preds:
        mae_test = np.asarray([metrics.mean_absolute_error(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
        r2_test = np.asarray([metrics.r2_score(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
        mse_test = np.asarray([metrics.mean_squared_error(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
        cor_test = np.asarray([np.corrcoef(y[test], preds[name]['y_test'][test])[0, 1] for train, test in cv.split(X, y)])
        size_test = np.asarray([len(y[test]) for train, test in cv.split(X, y)])

        stats_folds_list.append(
            np.concatenate((np.array([name] * len(mae_test))[:, None],
                            np.array(["CV%i" % cv for cv in range(len(mae_test))])[:, None],
                            mae_test[:, None], r2_test[:, None], mse_test[:, None], cor_test[:, None],
                            np.asarray(preds[name]['time'])[:, None],
                            np.asarray(preds[name]['best_param'])[:, None],
                            size_test[:, None]), axis=1))
        stats_list.append([name, np.mean(mae_test), np.mean(r2_test), np.mean(mse_test), np.mean(cor_test),
                           np.mean(preds[name]['time']), str(preds[name]['best_param'])])

        stats_df = pd.DataFrame(stats_list, columns=['model', 'mae_test', 'r2_test', 'mse_test', 'cor_test', "time", "best_param"])
        stats_df["size"] = len(y)

        stats_folds_df = pd.DataFrame(np.concatenate(stats_folds_list, axis=0),
                                      columns=['model', 'fold', 'mae_test', 'r2_test', 'mse_test', 'cor_test', "time", "best_param", "size_test"])

    model_params = None

    return stats_df, stats_folds_df, model_params
