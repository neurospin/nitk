#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:52:05 2020

@author: edouard.duchesnay@cea.fr
"""

from sklearn.externals.joblib import Parallel, delayed


def parallel(func, args_collection, n_jobs=5, pass_key=False, verbose=10):
    """Parallel execution of function `func` given argument `args_dict` dict

    Parameters
    ----------
    func: function

    args_collection: dict or list
        If dict, each (key, val) pair contains the key of the arguments given
        to to func.

    n_jobs: int
        Number of jobs

    pass_key: bool (default False)
        If true passes the key (as the first argument) to func. Use it to access
        to some global structure given the key.

    Return
    ------
    Collection of retun values of func. If args_collection is a dict, it
    returns a dictionary of (key, return value)

    Example
    -------
    >>> from nitk.utils import dict_product, parallel
    >>> # Prepare collection of arguments
    >>> args_collection = dict_product({"1":1, "2":2}, {"3":3, "4":4})
    >>> print(args_collection)
    {('1', '3'): [1, 3], ('1', '4'): [1, 4], ('2', '3'): [2, 3], ('2', '4'): [2, 4]}
    >>> def add(a, b):
    ...     return a + b
    >>> parallel(add, args_collection, n_jobs=5, verbose=0)
    {('1', '3'): 4, ('1', '4'): 5, ('2', '3'): 5, ('2', '4'): 6}
    >>> # Use key to access some global structure:
    >>> glob = {('1', '3'): -1, ('1', '4'): 1, ('2', '3'): -1, ('2', '4'): 1}
    >>> def add_glob(key, a, b):
    ...     return glob[key] * (a + b)
    >>> parallel(add_glob, args_collection, n_jobs=5, pass_key=True, verbose=0)
    {('1', '3'): -4, ('1', '4'): 5, ('2', '3'): -5, ('2', '4'): 6}
    """

    parallel_ = Parallel(n_jobs=n_jobs, verbose=verbose)

    if isinstance(args_collection, dict):
        def call(key, func, *args):
            if pass_key:
                return key, func(key, *args)
            else:
                return key, func(*args)

        cv_ret = parallel_(
            delayed(call)(key, func, *args) for key, args in args_collection.items())

        return {k: v for k, v in cv_ret}

    else:
        cv_ret = parallel_(delayed(func)(*args) for args in args_collection)

        return cv_ret
