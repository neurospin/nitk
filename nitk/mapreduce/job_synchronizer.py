#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:41:37 2020

@author: edouard.duchesnay@cea.fr

"""

import os
import time
from filelock import FileLock

class JobSynchronizer:

    """Job (Process) synchronizer based on distributed filesystem.
    set_state(self, key, state, previous_state=[]) control the state of a given
    job (identified with the key).
    Starting state is "INIT",

    First `start(key="0")` returns 'OK'
    Next call to to `start(key="0")` will returns the current job state: 'STARTED'
    or 'DONE'.

    Parameters
    ----------

    dir str
        The directory path shared to store log/lock files to synchronize the
        jobs.

    Example
    -------
    >>> from nitk.mapreduce import JobSynchronizer
    >>> import tempfile
    >>> js = JobSynchronizer(tempfile.mkdtemp(prefix="jobsynchro_"))

    >>> js.set_state(key="A", state="STARTED", previous_state=["INIT"])
    ('INIT', True)

    >>> js.set_state(key="A", state="STARTED", previous_state=["INIT"]) # Failed
    ('STARTED', False)

    >>> js.set_state(key="A", state="DONE", previous_state=["STARTED"])
    ('STARTED', True)

    >>> js.set_state(key="A", state="INIT") # Reset to INIT
    ('DONE', True)
    """
    def __init__(self, dir):
        self.dir = dir

    def key_to_filename(self, key):
        log_filename = os.path.join(self.dir,
                     'task_%s' % str(key))
        lock_filename = log_filename + ".lock"
        return log_filename, lock_filename

    def get_state(self, key):
        log_filename, lock_filename = self.key_to_filename(key)
        lock = FileLock(lock_filename)
        state = None
        with lock.acquire(timeout=30):

            # Get previous state, if log file doesn't exist: INIT
            if os.path.exists(log_filename):
                with open(log_filename, "r") as fd:
                    state = fd.readline().split(' ')[0]

        return state

    def set_state(self, key, state, previous_state=[]):
        """Set state of job identified by key.
        Next call to to `start(key="0")` will returns the current job state: 'STARTED'
        or 'DONE'.

        Parameters
        ----------
        key str
            The job unique key

        state str
            The state to set

        previous_state list
            list previous state of permitted state transition. Default is
            empty, no constraint. Not permitted transition wil return False
            as success.

        return
        ------
        string, boolean
            previous_state, success

        """
        log_filename, lock_filename = self.key_to_filename(key)
        lock = FileLock(lock_filename)
        with lock.acquire(timeout=30):

            # Get previous state, if log file doesn't exist: INIT
            if os.path.exists(log_filename):
                with open(log_filename, "r") as fd:
                    previous_state_ = fd.readline().split(' ')[0]
            else:
                previous_state_ = 'INIT'

            # Set state if transition allowed by previous_state
            if len(previous_state) == 0 or previous_state_ in previous_state:
                with open(log_filename, "w") as fd:
                    fd.write('%s %s %s' % (state, time.strftime("%Y-%m-%d-%H-%M-%S"), os.uname()[1]))
                return previous_state_, True
            else:
                return previous_state_, False


if __name__ == "__main__":

    import numpy as np
    from multiprocessing import Pool, Process, Manager
    import tempfile

    tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #####################################
    # Tasks are synchronized by the master

    def f(x):
        return x ** 2

    with Pool(5) as p:
        print(p.map(f, tasks))

    #########################################################################
    # Autonomous tasks: use the JobSynchronizer to execute tasks that has not
    # been done yet

    from nitk.utils import JobSynchronizer

    def apply_f(js, tasks, output):
        # iterate over tasks, try to start, if success process, then end;
        for x in tasks:
            if js.set_state(key=str(x), state="STARTED", previous_state=["INIT"])[1]:
                output.append(x * x)
                js.set_state(key=str(x), state="DONE", previous_state=["STARTED"])

        return output

    js = JobSynchronizer(tempfile.mkdtemp(prefix="jobsynchro_"))
    output_shared = Manager().list()

    pool = [Process(target=apply_f, args=(js, tasks, output_shared)) for i in range(2)]
    [j.start() for j in pool]
    [j.join() for j in pool]
    output = np.array(output_shared)
    print(output)

    output.sort()
    assert np.all(output ==  np.array(tasks) ** 2)
