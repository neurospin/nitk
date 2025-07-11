import time
from joblib import Parallel, delayed
from joblib import cpu_count

def run_sequential(func, iterable_dict, memory=None,  verbose=0,
                   *args, **kwargs):
    """Run a function sequentially over items in a dictionary.
    Uses a simple for loop to apply the function to each item in the dictionary.

    Parameters
    ----------
    func : callable
        The function to execute sequentially. It should accept the values from `iterable_dict` as arguments.
        The function signature should be `func(*args, verbose=0)`.
        The `verbose` argument is optional and can be used for logging.
        The function should return a result that will be collected in a dictionary.
    iterable_dict : dict
        Dictionary where each value is a tuple of arguments to pass to `func`.
    memory : object, optional
        Placeholder for memory caching (not used in this function), by default None.
    verbose : int, optional
        Verbosity level. If > 0, prints elapsed time, by default 0.
    *args
        Additional positional arguments to pass to `func`.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    _type_
        _description_
    """

    start_time = time.time()
    res = {k:func(*v, verbose=verbose) for k, v in iterable_dict.items()}

    if verbose > 0:
        print('Sequential execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))
    
    return res


def run_parallel(func, iterable_dict, memory=None, n_jobs=None, verbose=0,
                   *args, **kwargs):
    """
    Run a function in parallel over items in a dictionary.
    Uses Joblib's Parallel and delayed for parallel execution.

    Parameters
    ----------
    func : callable
        The function to execute in parallel.
        It should accept the values from `iterable_dict` as arguments.
        The function signature should be `func(*args, verbose=0)`.
        The `verbose` argument is optional and can be used for logging.
        The function should return a result that will be collected in a dictionary.
    iterable_dict : dict
        Dictionary where each value is a tuple of arguments to pass to `func`.
    memory : object, optional
        Placeholder for memory caching (not used in this function), by default None.
    n_jobs : int, optional
        Number of parallel jobs to run. If None, uses the number of physical CPU cores.
    verbose : int, optional
        Verbosity level. If > 0, prints elapsed time, by default 0.
    *args
        Additional positional arguments to pass to `func`.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    dict
        Dictionary mapping the original keys to the results returned by `func`.

    Example
    -------
    >>> iterable_dict = {'a': (1, 2), 'b': (3, 4)}
    >>> def func(x, y, verbose=0):
    ...     if verbose > 0:
    ...         print(f"Processing {x}, {y}")
    ...     return x + y
    >>> results = run_parallel(func, iterable_dict, n_jobs=2, verbose=)
    >>> print(results)
    Processing 1, 2
    Processing 3, 4
    {'a': 3, 'b': 7}
    """

    if not n_jobs:
        n_jobs = cpu_count(only_physical_cores=True)
        
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

    start_time = time.time()
    res = parallel(delayed(func)(*v, verbose=verbose)
                   for k, v in iterable_dict.items())

    if verbose > 0:
        print('Parallel execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))

    return {k:r for k, r in zip(iterable_dict.keys(), res)}

