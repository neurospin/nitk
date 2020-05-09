#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:52:05 2020

@author: edouard.duchesnay@cea.fr
"""

import os
from filelock import FileLock
from joblib import Parallel, delayed
import pickle
import re
import glob
from .job_synchronizer import JobSynchronizer

class MapReduce:
    """
    Parameters
    ----------

    n_jobs: int
        Number of jobs

    sync_dir str


    pass_key: bool (default False)
        If true passes the key (as the first argument) to func. Use it to access
        to some global structure given the key.

    verbose int
        verbose level

    Example
    -------

    >>> from nitk.utils import dict_product, parallel
    >>> # Prepare collection of arguments
    >>> key_values = dict_product({"1":1, "2":2}, {"3":3, "4":4})
    >>> print(key_values)
    {('1', '3'): [1, 3], ('1', '4'): [1, 4], ('2', '3'): [2, 3], ('2', '4'): [2, 4]}
    >>> def add(a, b):
    ...     return a + b
    >>> parallel(add, key_values, n_jobs=5, verbose=0)
    {('1', '3'): 4, ('1', '4'): 5, ('2', '3'): 5, ('2', '4'): 6}
    >>> # Use key to access some global structure:
    >>> glob = {('1', '3'): -1, ('1', '4'): 1, ('2', '3'): -1, ('2', '4'): 1}
    >>> def add_glob(key, a, b):
    ...     return glob[key] * (a + b)
    >>> parallel(add_glob, key_values, n_jobs=5, pass_key=True, verbose=0)
    {('1', '3'): -4, ('1', '4'): 5, ('2', '3'): -5, ('2', '4'): 6}
    """

    def __init__(self, n_jobs=5, sync_dir=None, pass_key=False, verbose=10):

        self.n_jobs = n_jobs
        self.sync_dir = sync_dir
        self.pass_key = pass_key
        self.verbose = verbose


    def map(self, func, key_values):

        return self.map_centralized(func, key_values) if self.sync_dir is None \
            else self.map_distributed(func, key_values)

    def map_centralized(self, func, key_values):
        """Parallel execution of function `func` given argument `args_dict` dict

        Parameters
        ----------

        func: function

        key_values: dict or list
            if dict: dict of (key, val) pairs. Where val are func input arguments
            if list: list val. Where val are func input arguments

        Return
        ------
        dict of (key, val) pairs. val are output of func.
        """

        parallel_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        if isinstance(key_values, dict):
            def call(key, func, *args):
                if self.pass_key:
                    return key, func(key, *args)
                else:
                    return key, func(*args)

            cv_ret = parallel_(
                delayed(call)(key, func, *args) for key, args in key_values.items())

            return {k: v for k, v in cv_ret}

        else:
            cv_ret = parallel_(delayed(func)(*args) for args in key_values)

            return cv_ret

    def map_distributed(self, func, key_values):
        """ Distributed execution of parallel based on filesystem synchronisation
        to track execution state.
        Multiple instances of "parallel" can be executed on different computer.
        """

        self.n_tasks = len(key_values)

        parallel_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        js = JobSynchronizer(self.sync_dir)

        def call(key, js, func, *args):

            if js.set_state(key=str(key), state="STARTED", previous_state=["INIT"])[1]:
                ret_ =  func(key, *args) if self.pass_key else func(*args)
                # ret_["__key__"] = key
                js.set_state(key=str(key), state="DONE", previous_state=["STARTED"])
                pickle_filename = js.key_to_filename(str(key))[0] + ".pkl"
                # print(key, ret_, "=>", pickle_filename)
                with open(pickle_filename, 'wb') as fd:
                    pickle.dump(ret_, fd)

            else:
                ret_ = 0

            return key, ret_

        cv_ret = parallel_(
            delayed(call)(key, js, func, *args) for key, args in key_values.items())

        return {k: v for k, v in cv_ret if v is not None}

    def reduce_load_results(self):

        re_key = re.compile('task_([^\.pkl]*)')
        res = dict()
        map_ouptut_filenames = glob.glob(os.path.join(self.sync_dir, "task_*.pkl"))

        if len(map_ouptut_filenames) == self.n_tasks:

            for f in map_ouptut_filenames:
                key_ = eval(re_key.findall(f)[0])
                with open(f, 'rb') as fd:
                    res_ = pickle.load(fd)
                res[key_] = res_

            return res

        else:

            return None


def parallel(func, key_values, n_jobs=5, pass_key=False, verbose=10):
    """ Helper function for  MapReduce(...).map()

    Parameters
    ----------
    func : function
        DESCRIPTION.
    key_values : dict
        DESCRIPTION.
    n_jobs : int, optional
        DESCRIPTION. The default is 5.
    pass_key : bool, optional
        DESCRIPTION. The default is False.
    verbose : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mp = MapReduce(n_jobs=n_jobs, pass_key=pass_key, verbose=verbose)
    return mp.map(func=func, key_values=key_values)


if __name__ == "__main__":

    from nitk.utils import dict_product, MapReduce
    import numpy as np
    # Prepare collection of arguments


    N = 10
    key_values = dict_product({str(v):v for v in np.arange(N)},
                              {str(v):v for v in (np.arange(N) * 10)})
    def add(a, b):
        import time
        time.sleep(1)
        return a + b

    print("##################################################################")
    print("Centralized Mapper")

    res_single = MapReduce(n_jobs=5, verbose=20).map(add, key_values)
    print(res_single)

    print("##################################################################")
    print("Distributed Mapper")

    import tempfile
    import os
    import shutil

    sync_dir = os.path.join(tempfile.gettempdir(), "mapreduce_add")
    shutil.rmtree(sync_dir)
    os.makedirs(sync_dir)
    print("sync_dir:", sync_dir)

    print("## Run mapper ##")
    mp = MapReduce(n_jobs=5, sync_dir=sync_dir, verbose=20)
    mp.map(add, key_values)

    print("## Run reducer ##")
    res_multi = mp.reduce_load_results()

    if res_multi is not None:
        print("All task completed could be loaded")
        print(res_multi)
        print(res_multi == res_single)
    else:
        print("Some tasks were not finished aborted redcue")


