# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 17:39:34 2014

@author: edouard.duchesnay@cea.fr
"""

import numpy as np

def arr_get_threshold_from_norm2_ratio(v, ratio=.99):
    """Get threshold to apply to a 1d array such
    ||v[np.abs(v) >= t]|| / ||v|| == ratio
    return the threshold.

    Example
    -------
    >>> import numpy as np
    >>> v = np.random.randn(1e6)
    >>> t = arr_get_threshold_from_norm2_ratio(v, ratio=.5)
    >>> v_t = v.copy()
    >>> v_t[np.abs(v) < t] = 0
    >>> ratio = np.sqrt(np.sum(v[np.abs(v) >= t] ** 2)) / np.sqrt(np.sum(v ** 2))
    >>> print np.allclose(ratio, 0.5)
    True
    """
    #shape = v.shape
    import numpy as np
    v = v.copy().ravel()
    v2 = (v ** 2)
    v2.sort()
    v2 = v2[::-1]
    v_n2 = np.sqrt(np.sum(v2))
    #(v_n2 * ratio) ** 2
    cumsum2 = np.cumsum(v2)  #np.sqrt(np.cumsum(v2))
    select = cumsum2 <= ((v_n2 * ratio) ** 2)
    if select.sum() !=0:
        thres = np.sqrt(v2[select][-1])

    else:
        thres = 0

    return thres

def arr_threshold_from_norm2_ratio(v, ratio=.99):
    """Threshold input array such
    ||v[np.abs(v) >= t]|| / ||v|| == ratio
    return the thresholded vector and the threshold

    Example
    -------
    >>> import numpy as np
    >>> v = np.random.randn(1e6)
    >>> v_t, t = arr_threshold_from_norm2_ratio(v, ratio=.5)
    >>> ratio = np.sqrt(np.sum(v_t ** 2)) / np.sqrt(np.sum(v ** 2))
    >>> print np.allclose(ratio, 0.5)
    """
    t = arr_get_threshold_from_norm2_ratio(v, ratio=ratio)
    v_t = v.copy()
    v_t[np.abs(v) < t] = 0
    return v_t, t

def maps_similarity(maps):
    """Map's measures of similarity

    Parameters
    ----------
    maps : array(N, P)
        compute similarity measures between N maps of dimension P.

    Returns
    -------
    r_bar: average pairwize corelation,
    dice_bar: average pairwize dice index
    fleiss_kappa_stat:

    """
    from statsmodels.stats.inter_rater import fleiss_kappa

    # Correlation
    R = np.corrcoef(maps)
    R = R[np.triu_indices_from(R, 1)]
    # Fisher z-transformation / average
    z_bar = np.mean(1. / 2. * np.log((1 + R) / (1 - R)))
    # bracktransform
    r_bar = (np.exp(2 * z_bar) - 1) / (np.exp(2 * z_bar) + 1)

    maps_sign = np.sign(maps)

    # Paire-wise Dice coeficient
    try:
        ij = [[i, j] for i in range(maps.shape[0]) for j in range(i+1, maps.shape[0])]
        dices = list()
        for idx in ij:
            A, B = maps_sign[idx[0], :], maps_sign[idx[1], :]
            dices.append(float(np.sum((A == B)[(A != 0) & (B != 0)])) / (np.sum(A != 0) + np.sum(B != 0)))
        dice_bar = np.mean(dices)
    except:
        dice_bar = np.NaN

    try:
        # Compute Fleiss-Kappa statistics
        table = np.zeros((maps_sign.shape[1], 3))
        table[:, 0] = np.sum(maps_sign == 0, 0)
        table[:, 1] = np.sum(maps_sign == 1, 0)
        table[:, 2] = np.sum(maps_sign == -1, 0)
        fleiss_kappa_stat = fleiss_kappa(table)
    except:
        fleiss_kappa_stat = np.NaN

    return r_bar, dice_bar, fleiss_kappa_stat

"""
np.save("/tmp/betas.npy" ,betas)

betas = np.load("/tmp/betas.npy")
v = betas[1, :]
v_t, t = arr_threshold_from_norm2_ratio(v, ratio=.99)
ratio = np.sqrt(np.sum(v_t ** 2)) / np.sqrt(np.sum(v ** 2))
from brainomics import array_utils

betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in xrange(betas.shape[0])])
assert np.allclose(np.sqrt(np.sum(betas_t ** 2, 1)) /
                np.sqrt(np.sum(betas ** 2, 1)), [0.99]*5)
"""