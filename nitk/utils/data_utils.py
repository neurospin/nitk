#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:53:57 2020

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import pandas as pd
from itertools import product


def dict_product(*dicts):
    """Cartesian product of 2 dictionnaries: (keys x keys) (values )

    Example
    -------
    >>> dict_product({"1":1, "2":2}, {"3":3, "4":4}, {"A":"A", "B":"B"})
    {('1', '3', 'A'): [1, 3, 'A'], ('1', '3', 'B'): [1, 3, 'B'], ('1', '4', 'A'): [1, 4, 'A'], ('1', '4', 'B'): [1, 4, 'B'], ('2', '3', 'A'): [2, 3, 'A'], ('2', '3', 'B'): [2, 3, 'B'], ('2', '4', 'A'): [2, 4, 'A'], ('2', '4', 'B'): [2, 4, 'B']}
    """
    return {keys:[dicts[i][k] for i, k in enumerate(keys)] for keys in product(*dicts)}


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
                aggregate_preds[aggregate_key][test] = pred[ret_key]

    return aggregate_preds
