#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:53:57 2020

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def dict_product(*dicts):
    """Cartesian product of 2 dictionnaries: (keys x keys) (values )

    Example
    -------
    >>> dict_product({"1":1, "2":2}, {"3":3, "4":4}, {"A":"A", "B":"B"})
    {('1', '3', 'A'): [1, 3, 'A'], ('1', '3', 'B'): [1, 3, 'B'], ('1', '4', 'A'): [1, 4, 'A'], ('1', '4', 'B'): [1, 4, 'B'], ('2', '3', 'A'): [2, 3, 'A'], ('2', '3', 'B'): [2, 3, 'B'], ('2', '4', 'A'): [2, 4, 'A'], ('2', '4', 'B'): [2, 4, 'B']}
    """
    return {keys:[dicts[i][k] for i, k in enumerate(keys)] for keys in product(*dicts)}



def reduce_cv_classif(key_vals, cv_dict, y_true, index_fold=-1):
    """ Reduce function take runs key/val and CV metrics for classification

    Parameters
    ----------
    key_vals dict
        dict of key/val for each run where
            key=run_key, val=run_predictions
            - run_key  ::= (<param_0>, ..., <param_p>, <fold>)
            - run_predictions ::= dict(key=pred_key, val=prediction))

    cv_dict dict
        dict(key<fold_key>, val=[train_idx, test_idx])

    y_true array

    Return
    ------
        pd.DataFrame

    Example
    -------
    >>> y_true = np.array([0, 1, 0, 1, 1])
    >>> cv_dict = dict(cv0=[[0, 1], [2, 3, 4]], cv1=[[2, 3, 4], [0, 1]])
    >>> key_vals = {
          ('a', 'cv0'):dict(y_test=np.array([0, 1, 1]), score_test=np.array([.1, .8, .7])),
          ('a', 'cv1'):dict(y_test=np.array([0, 1]), score_test=np.array([.1, .7])),
          ('b', 'cv0'):dict(y_test=np.array([0, 1, 0]), score_test=np.array([.1, .6, .2])),
          ('b', 'cv1'):dict(y_test=np.array([0, 0]), score_test=np.array([.1, .3]))}
    >>> reduce_cv_classif(key_vals, cv_dict, y_true)
      param_0 fold  pred  auc  bacc  recall_0  recall_1  count_0  count_1
    0       a  cv0  test  1.0  1.00       1.0       1.0        1        2
    1       a  cv1  test  1.0  1.00       1.0       1.0        1        1
    2       b  cv0  test  1.0  0.75       1.0       0.5        1        2
    3       b  cv1  test  1.0  0.50       1.0       0.0        1        1
    """
    labels = np.unique(y_true)
    is_binary = True if len(labels) == 2 else False
    res = list()
    # Iterate over run
    for run_key, run_val in key_vals.items():
        # DEBUG break
        fold = run_key[index_fold] # last key match the fold
        train, test = cv_dict[fold]
        # Iterate over predicted values of  given run
        for pred_key, pred_val in select_predictions(run_val).items():
            # DEBUG pred_key, pred_val = list(select_predictions(run_val).items())[1]
            y_true_ = y_true[test] if pred_key[0] == 'test' else y_true[train]
            _, recall, _, count = precision_recall_fscore_support(y_true_, pred_val[0].ravel(), labels=labels,  warn_for='recall')
            bacc = np.mean(recall[count > 0])
            auc = roc_auc_score(y_true_, pred_val[1].ravel()) if len(pred_val) == 2 and is_binary else np.nan
            res.append(
                list(run_key) + ["_".join(pred_key)] + [auc, bacc] + recall.tolist() + count.tolist())

    # columns =  ["param_%i" % p for p in range(len(run_key) - 1)] + ["fold"] + ["pred"] +\
    #        ["auc", "bacc"] + ["recall_%i" % lab for lab in labels] + \
    #       ["count_%i" % lab for lab in labels]


    columns =  ["param_%i" % p for p in range(len(run_key))] + ["pred"] +\
            ["auc", "bacc"] + ["recall_%i" % lab for lab in labels] + \
            ["count_%i" % lab for lab in labels]

    if index_fold == -1:
        columns[len(run_key)-1] = "fold"
    else:
        columns[index_fold] = "fold"

    return pd.DataFrame(res, columns=columns)

def select_predictions(ret_dict):
    """ Select prediction according to the pattern <y|score>_<train|test>[<_suffix>].
    And match label with score, ie,
    Match <y>_<train|test>[<_suffix>] with corresponding <score>_<train|test>[<_suffix>]

    Parameters
    ----------
        ret_dict dict
            {<y|score>_<train|test>[<_suffix>] : pred_arr}
    Return
    ------
        dict {(<train|test>, [<suffix>]) : [y, [score]]}

    Example
    -------
    >>> ret_dict = dict(y_test=[0, 0], score_test=[.1, .1], y_test_im=[1, 1], score_test_im=[.9, .9])
    >>> select_predictions(ret_dict)
    {('test',): [[0, 0], [0.1, 0.1]], ('test', 'im'): [[1, 1], [0.9, 0.9]]}
    """
    ret_keys_ = {tuple(k.split('_')):k for k in ret_dict.keys()}
    pred_dict = dict() # {(test|train, [suffix]): [y_name, score_name]}
    for k, var in ret_keys_.items():
        if k[0] == 'y':
            pred_dict[k[1:]] = [ret_dict[var]]
            score_matched_key = tuple(["score"] + list(k[1:]))
            if score_matched_key in ret_keys_:
                pred_dict[k[1:]].append(ret_dict[ret_keys_[score_matched_key]])
    return pred_dict


def group_by(dict_, grp_keys_idx):
    """
    Split dict_ in dict of dict, where inner dicts are groups

    Parameters
    ----------
    dict_: dict where keys are tubles
        keys: parameters configuration (algo1, CV0) (algo1, CV1) (algo2, CV0)
        values: dict(y_test=preds, score_test=preds) dict(y_test=preds, score_test=preds)

    grp_keys_idx: list of int
        indices in keys

    Return
    ------
        dict of dict

    Example
    -------
    >>> dict_ = {("a", "cv0"): [1, 2], ("a", "cv1"):[3, 4],
                ("b", "cv0"): [10, 20], ("b", "cv0"):[30, 40]}
    >>> group_by(dict_, [0])
    {'a': {'cv0': [1, 2], 'cv1': [3, 4]}, 'b': {'cv0': [30, 40]}}
    """
    grp_keys_idx = tuple(grp_keys_idx)
    other_keys_idx = tuple(set(list(range(len(list(dict_.keys())[0])))) - set(grp_keys_idx))

    grps = dict()
    for key, val in dict_.items():
        grp_keys = tuple([key[idx] for idx in grp_keys_idx]) if len(grp_keys_idx) > 1 else key[grp_keys_idx[0]]
        inner_keys = tuple([key[i] for i in other_keys]) if len(other_keys) > 1 else key[other_keys[0]]
        if not grp_keys in grps:
            grps[grp_keys] = {inner_keys:val}
        else:
            grps[grp_keys][inner_keys] = val

    return grps


def aggregate_cv(cv_res, args_collection, cv_idx):
    """ Aggregate Cross-validation results: Given a dictionnary of predictions

    Parameters
    ----------
    cv_res: dict
        keys: parameters configuration (algo1, CV0) (algo1, CV1) (algo2, CV0)
        values: dict(y_test=preds, score_test=preds) dict(y_test=preds, score_test=preds)

    args_collection: dict
        keys: parameters configuration (algo1, CV0) (algo1, CV1) (algo2, CV0)
        values: (estimator1, [train, test]) (estimator1, [train, test]) ...

    cv_idx: int
        the index in keys and values of the Cross-validation here 1

    Return
    ------
        dict of aggregated prediction over CV

    Example
    -------
    from nitk.utils import aggregate_cv
    # test predictions for algo in {a, b} and fold in {0, 1}
    cv_res = {('a', 0):dict(y_test=np.array([1, 1]), score_test=np.array([0.9, 0.8])),
              ('a', 1):dict(y_test=np.array([0]), score_test=np.array([0.1])),
              ('b', 0):dict(y_test=np.array([1, 1]), score_test=np.array([0.7, 0.6])),
              ('b', 1):dict(y_test=np.array([0]), score_test=np.array([0.3]))}

    # arguments algo in {a, b} and fold in {0, 1} where splits indices (of size (1, 2) and (2, 1))
    args_collection = {('a', 0):['a', [[0],    [1, 2]]],
                       ('a', 1):['a', [[1, 2], [0]]],
                       ('b', 0):['b', [[0],    [1, 2]]],
                       ('b', 1):['b', [[1, 2], [0]]]}

    aggregate_cv(cv_res, args_collection, cv_idx=1)
    {('a', 'score_test'): array([0.1, 0.9, 0.8]),
     ('a', 'y_test'): array([0., 1., 1.]),
     ('b', 'score_test'): array([0.3, 0.7, 0.6]),
     ('b', 'y_test'): array([0., 1., 1.])}
    """
    print("DEPRECATED")
    # aggregated result keys = (cv_res - key at cv_idx) x return_keys
    # key_without_cv = np.delete(np.array([k for k in cv_res.keys()]), cv_idx, axis=1).tolist()
    d = pd.DataFrame([k for k in cv_res.keys()]).drop(cv_idx, axis=1).drop_duplicates()
    key_without_cv = [list(r) for r in d.itertuples(index=False)]

    return_keys = list(np.unique(np.array([list(pred.keys()) for pred in cv_res.values()])))

    import itertools
    aggregate_keys = [tuple(a + [b]) for a, b in itertools.product(key_without_cv, return_keys)]
    pred_len = np.unique([np.sum([len(split) for split in v[cv_idx]]) for v in args_collection.values()])
    assert len(pred_len) == 1
    pred_len = pred_len[0]

    aggregate_preds = {k:np.zeros(pred_len) for k in aggregate_keys}

    for param_key, pred in cv_res.items():
        # print(param_key, pred)
        train, test = args_collection[param_key][cv_idx]
        param_key = list(param_key)
        param_key.pop(cv_idx)

        for ret_key in pred.keys():
            aggregate_key = tuple(param_key + [ret_key])
            if isinstance(pred[ret_key], np.ndarray) and len(pred[ret_key]) == len(test):
                aggregate_preds[aggregate_key][test] = pred[ret_key].ravel()

    return aggregate_preds
