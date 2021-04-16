#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:49:53 2021

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import nibabel

def vec_to_img(vec, mask_img):
    """Flat vector to nii image, where values


    Parameters
    ----------
    vec : flat vector
        vector of values within.
    mask_img : nii
        Boolean mask, mask_arr.sum() == len(vec).

    Returns
    -------
    nii image similar to mask_img.

    """
    mask_arr = mask_img.get_fdata() != 0
    mask_arr.sum() == len(vec), "Missmatch between mask and flat vector"
    val_arr = np.zeros(mask_img.shape)
    val_arr[mask_arr] = vec
    return nibabel.Nifti1Image(val_arr, affine=mask_img.affine)
