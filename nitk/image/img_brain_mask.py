#!/usr/bin/env python3
# -*- coding: utf-8 -*-
epilog = """
Created on Wed Feb  5 15:29:39 2020

@author: edouard.duchesnay@cea.fr

Compute brain mask

Example:
python nitk/image/img_brain_mask.py --input /neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii
"""

import numpy as np
import scipy
import pandas as pd
import nibabel
import argparse
from  nitk.bids import get_keys
from  nitk.image import img_to_array

import nilearn
import nilearn.masking


def compute_brain_mask(imgs, target_img=None, mask_thres_mean=0.1, mask_thres_std=1e-6, clust_size_thres=10, verbose=1):
    """
    Compute brain mask:
    (1) Implicit mask threshold `mean >= mask_thres_mean` and `std >= mask_thres_std`
    (2) Use brain mask from `nilearn.masking.compute_gray_matter_mask(target_img)`
    (3) mask = Implicit mask & brain mask
    (4) Remove small branches with `scipy.ndimage.binary_opening`
    (5) Avoid isolated clusters: remove clusters (of connected voxels) smaller that `clust_size_thres`

    Parameters
    ----------
    imgs : [str] path to images
        or array (n_subjects, 1, , image_axis0, image_axis1, ...) in this case
        target_img must be provided.

    target_img : nii image
        Image defining the referential.

    mask_thres_mean : float (default 0.1)
        Implicit mask threshold `mean >= mask_thres_mean`

    mask_thres_std : float (default 1e-6)
        Implicit mask threshold `std >= mask_thres_std`

    clust_size_thres : float (clust_size_thres 10)
        Remove clusters (of connected voxels) smaller that `clust_size_thres`

    verbose : int (default 1)
        verbosity level

    expected : dict
        optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
         nii image:
             In referencial of target_img or the first imgs

    Example
    -------
    Parameters
    ----------
    NI_arr :  ndarray, of shape (n_subjects, 1, image_shape).
    target_img : image.
    mask_thres_mean : Implicit mask threshold `mean >= mask_thres_mean`
    mask_thres_std : Implicit mask threshold `std >= mask_thres_std`
    clust_size_thres : remove clusters (of connected voxels) smaller that `clust_size_thres`
    verbose : int. verbosity level

    Returns
    -------
    image of mask

    Example
    -------
    >>> from  nitk.image import compute_brain_mask
    >>> import glob
    >>> imgs = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii")
    >>> mask_img = compute_brain_mask(imgs)
    Clusters of connected voxels #3, sizes= [368569, 45, 19]
    >>> mask_img.to_filename("/tmp/mask.nii")
    """

    if isinstance(imgs, list) and len(imgs) >= 1 and isinstance(imgs[0], str):
        imgs_arr, df, target_img = img_to_array(imgs)

    elif isinstance(imgs, np.ndarray) and imgs.ndim >= 5:
        imgs_arr = imgs
        assert isinstance(target_img, nibabel.nifti1.Nifti1Image)

    # (1) Implicit mask
    mask_arr = np.ones(imgs_arr.shape[1:], dtype=bool).squeeze()
    if mask_thres_mean is not None:
        mask_arr = mask_arr & (np.abs(np.mean(imgs_arr, axis=0)) >= mask_thres_mean).squeeze()
    if mask_thres_std is not None:
        mask_arr = mask_arr & (np.std(imgs_arr, axis=0) >= mask_thres_std).squeeze()

    # (2) Brain mask: Compute a mask corresponding to the gray matter part of the brain.
    # The gray matter part is calculated through the resampling of MNI152 template
    # gray matter mask onto the target image
    # In reality in is a brain mask
    mask_img = nilearn.masking.compute_gray_matter_mask(target_img)

    # (3) mask = Implicit mask & brain mask
    mask_arr = (mask_img.get_data() == 1) & mask_arr

    # (4) Remove small branches
    mask_arr = scipy.ndimage.binary_opening(mask_arr)

    # (5) Avoid isolated clusters: remove all cluster smaller that clust_size_thres
    mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)

    labels = np.unique(mask_clustlabels_arr)[1:]
    for lab in labels:
        clust_size = np.sum(mask_clustlabels_arr == lab)
        if clust_size <= clust_size_thres:
            mask_arr[mask_clustlabels_arr == lab] = False

    if verbose >= 1:
        mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
        labels = np.unique(mask_clustlabels_arr)[1:]
        print("Clusters of connected voxels #%i, sizes=" % len(labels),
              [np.sum(mask_clustlabels_arr == lab) for lab in labels])

    return nilearn.image.new_img_like(target_img, mask_arr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('--input', help='list of niftii images', nargs='+', type=str)
    parser.add_argument('-o', '--output', help='niftii file for the mask', type=str)
    options = parser.parse_args()

    # TODO extends with additional parameters

    if options.input is None:
        parser.print_help()
        raise SystemExit("Error: Input is missing.")

    if options.output is None:
        options.output = "mask.nii.gz"

    mask_img = compute_brain_mask(options.input)
    mask_img.to_filename(options.output)
