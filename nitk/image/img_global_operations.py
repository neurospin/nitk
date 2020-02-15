#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 01:30:25 2020

@author: edouard.duchesnay@cea.fr
"""

import numpy as np

def global_scaling(imgs_arr, axis0_values=None, target=1500):
    """
    Apply a global proportional scaling, such that axis0_values * gscaling == target

    Parameters
    ----------
    imgs_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    axis0_values: 1-d array, if None (default) use global average per subject: imgs_arr.mean(axis=1)
    target: scalar, the desired target

    Returns
    -------
    The scaled array

    >>> import numpy as np
    >>> from nitk.image import global_scaling
    >>> imgs_arr = np.array([[9., 11], [0, 2],  [4, 6]])
    >>> imgs_arr
    array([[ 9., 11.],
           [ 0.,  2.],
           [ 4.,  6.]])
    >>> axis0_values = [10, 1, 5]
    >>> global_scaling(imgs_arr, axis0_values, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    >>> global_scaling(imgs_arr, axis0_values=None, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    """
    if axis0_values is None:
        axis0_values = imgs_arr.mean(axis=1)
    gscaling = target / np.asarray(axis0_values)
    gscaling = gscaling.reshape([gscaling.shape[0]] + [1] * (imgs_arr.ndim - 1))
    return gscaling * imgs_arr


def center_by_site(imgs_arr, site, in_place=False):
    """
    Center by site

    Parameters
    ----------
    
    imgs_arr :  ndarray, of shape (n_subjects, 1, image_shape).

    site : 1-d array of site labels

    in_place: boolean perform inplace operation

    Returns
    -------
    
    imgs_arr
    
    Examples
    --------
    
    >>> import numpy as np
    >>> from nitk.image import center_by_site
    >>> imgs_arr = np.array([[8., 10], [9, 14],  [3, 5], [4, 7]])
    >>> imgs_arr
    array([[ 8., 10.],
           [ 9., 14.],
           [ 3.,  5.],
           [ 4.,  7.]])
    >>> preproc.center_by_site(imgs_arr, site=[1, 1, 0, 0])
    array([[-0.5, -2. ],
           [ 0.5,  2. ],
           [-0.5, -1. ],
           [ 0.5,  1. ]])
    """
    if not in_place:
        imgs_arr = imgs_arr.copy()
    site = np.asarray(site)
    for s in set(site):
        # s = 1
        m = site == s
        imgs_arr[m] -= imgs_arr[m, :].mean(axis=0)

    return imgs_arr

