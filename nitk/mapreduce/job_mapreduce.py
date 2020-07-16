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

    shared_dir str (optional) for map_distributed only
        A shared directory to store map outputs, log and lock files used to
        synchonize jobs.

    pass_key: bool (default False)
        If true passes the key (as the first argument) to func. Use it to access
        to some global structure given the key.

    verbose int
        verbose level

    Example
    -------

    >>> from nitk.mapreduce import dict_product, MapReduce, parallel
    >>> # Prepare collection of arguments
    >>> key_values = dict_product({"1":1, "2":2}, {"3":3, "4":4})
    >>> print(key_values)
    {('1', '3'): [1, 3], ('1', '4'): [1, 4], ('2', '3'): [2, 3], ('2', '4'): [2, 4]}
    >>> def add(a, b):
    ...     return a + b
    >>> MapReduce(n_jobs=5, verbose=0).map(add, key_values)
    {('1', '3'): 4, ('1', '4'): 5, ('2', '3'): 5, ('2', '4'): 6}
    >>> # Use helper function
    >>> parallel(add, key_values, n_jobs=5, verbose=0)
    {('1', '3'): 4, ('1', '4'): 5, ('2', '3'): 5, ('2', '4'): 6}
    """

    def __init__(self, n_jobs=5, shared_dir=None, pass_key=False, verbose=10):

        self.n_jobs = n_jobs
        self.shared_dir = shared_dir
        self.pass_key = pass_key
        self.verbose = verbose


    def map(self, func, key_values):

        return self.map_centralized(func, key_values) if self.shared_dir is None \
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
        """ Distributed execution of map based on filesystem synchronisation
        to track execution state.
        Multiple instances of the mapper can be executed on different computer.
        """

        self.n_tasks = len(key_values)

        parallel_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        js = JobSynchronizer(self.shared_dir)

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

    def reduce_collect_outputs(self):
        """Collect output key/value pairs produced by `map_distributed()` into
        a single dict. Return None if some expected pair were not available.
        Each discrtibuted instance of the mapper can safely call
        `reduce_collect_outputs()` only the last to finish will return the
        dictionary.

        Returns
        -------
        res : dict or None
            Return output key/value pairs or None if some are missing.

        """

        re_key = re.compile('^task_([^$]*)')
        # Fetch [key, filename] pairs in task_*.pkl"
        key_filenames_output = [[re_key.findall(os.path.splitext(os.path.basename(f))[0]),
                                 f] for f in glob.glob(os.path.join(self.shared_dir, "task_*.pkl"))]
        # Keep exactly those with one key
        key_filenames_output = [[pair[0][0], pair[1]] for pair in
                                key_filenames_output if len(pair[0]) == 1]

        key_values_output = dict()
        if len(key_filenames_output) == self.n_tasks:

            for key_str, value_filename in key_filenames_output:
                pass
                key = eval(key_str)
                with open(value_filename, 'rb') as fd:
                    value = pickle.load(fd)
                key_values_output[key] = value

            return key_values_output

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

    shared_dir = os.path.join(tempfile.gettempdir(), "mapreduce_add")
    shutil.rmtree(shared_dir)
    os.makedirs(shared_dir)
    print("shared_dir:", shared_dir)

    print("## Run mapper ##")
    mp = MapReduce(n_jobs=5, shared_dir=shared_dir, verbose=20)
    mp.map(add, key_values)

    print("## Run reducer ##")
    res_multi = mp.reduce_load_results()

    if res_multi is not None:
        print("All task completed could be loaded")
        print(res_multi)
        print(res_multi == res_single)
    else:
        print("Some tasks were not finished aborted redcue")


