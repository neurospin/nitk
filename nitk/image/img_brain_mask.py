#!/usr/bin/env python3
# -*- coding: utf-8 -*-
epilog = """
Created on Wed Feb  5 15:29:39 2020

@author: edouard.duchesnay@cea.fr

Compute brain mask

Example:
python nitk/image/img_brain_mask.py --input /neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii
"""
import os
import numpy as np
import scipy
import pandas as pd
import nibabel
import argparse

import nilearn
import nilearn.masking
from nilearn.image import resample_to_img

from  nitk.bids import get_keys
from  nitk.image import img_to_array
from  nitk.atlases import fetch_atlas_lobes


def rm_small_clusters(mask_arr, clust_size_thres=None):
    """ Remove clusters smaller than clust_size_thres

    Parameters
    ----------
    mask_arr : 3D array
        3D mask.
    clust_size_thres : int, optional
        The default is None: keep only the largest cluster

    Returns
    -------
    mask_arr : 3D array
        cluster smaller than clust_size_thres are removed.
    """
    mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)

    labels = np.unique(mask_clustlabels_arr)[1:]

    if clust_size_thres is None:
        clust_size_thres = [np.sum(mask_clustlabels_arr == lab) for lab in labels][0]

    for lab in labels:
        clust_size = np.sum(mask_clustlabels_arr == lab)
        if clust_size < clust_size_thres:
            mask_arr[mask_clustlabels_arr == lab] = False

    return mask_arr

def compute_brain_mask(target_img=None, imgs=None, implicitmask_arr=None,
                       tissue_prior="gray", mask_thres_mean=0.1,
                       mask_thres_std=1e-6, clust_size_thres=None,
                       rm_brainstem=False, rm_cerebellum=False,
                       fsl_home="/usr/share/fsl", verbose=0):
    """
    Compute brain mask:
    # (1) Tissue mask from priors
    # (2) Compute implicit mask if data are provided (optional)
    # (3) Apply Implicit mask if provided or computed from the data (optional)
    # (4) Remove Brain-Stem and Cerebellum (optional)
    # (5) Remove small branches
    # (6) Avoid isolated clusters: remove all cluster smaller that clust_size_thres

    Parameters
    ----------
    target_img : nii image (optional if imgs is provided)
        Image defining the referential.

    imgs : [str] optional.
        Path to images to compute implicite mask. if path are BIDS compatible
        participant_id will be returned.
        or array (n_subjects, 1, , image_axis0, image_axis1, ...) in this case
        target_img must be provided. If data are provided, it replaces the
        implicitmask_arr.

    implicitmask_arr : 3D array
        Implicit mask. If missing it is computed from imgs. If neither
        data "imgs" nor implicitmask_arr are provided mask will depend only
        on tissue priors.

    tissue_prior : str
        Tissue mask in "gray", "white", "brain". Threshold fsl priors at 0.2

    mask_thres_mean : float (default 0.1)
        Implicit mask threshold `mean >= mask_thres_mean`

    mask_thres_std : float (default 1e-6)
        Implicit mask threshold `std >= mask_thres_std`

    clust_size_thres : float (clust_size_thres None)
        Remove clusters (of connected voxels) smaller that `clust_size_thres`.
        If None keep only the largest cluster.

    rm_brainstem : bool
        Remove Brain-Steam ? (default False)

    rm_cerebellum : bool
        Remove cerebellum ? (default False)

    fsl_home : str
        (default "/usr/share/fsl")

    verbose : int (default 1)
        verbosity level


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
    >>> import numpy as np
    >>> import nibabel
    >>> from nilearn.image import resample_to_img
    >>> from  nitk.image import compute_brain_mask
    >>>
    >>> # 1) No implicit mask
    >>> target_img = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
    >>> mask_img = compute_brain_mask(target_img=target_img, verbose=False)
    >>> mask_img.to_filename("/tmp/mask_noimplicit.nii.gz")
    >>> mask_img = compute_brain_mask(target_img=target_img, rm_brainstem=True, rm_cerebellum=True)
    >>> mask_img.to_filename("/tmp/mask_noimplicit_nocereb-bs.nii.gz")
    >>>
    >>> # 2) Implicit mask from data (using 1 image !)
    >>> # provide images path like that:
    >>> # imgs = [glob.glob(".../bids_study/derivatives/sub-*/ses-*/mri/mwp1sub*.nii")]
    >>> sub_img = resample_to_img("/usr/share/fsl/data/standard/tissuepriors/avg152T1_gray.img", target_img)
    >>> sub_img = nibabel.Nifti1Image(sub_img.get_fdata().squeeze(), affine=sub_img.affine)
    >>> sub_img.to_filename("/tmp/sub-01.nii.gz")
    >>> imgs = ["/tmp/sub-01.nii.gz"]
    >>>
    >>> mask_img = compute_brain_mask(target_img=target_img, imgs=imgs, mask_thres_mean=0.1,
                                      mask_thres_std=0)
    >>> mask_img.to_filename("/tmp/mask_implicit.nii.gz")
    >>> # 3) Pre-computed implicit mask from data (using 1 image !)
    >>> implicitmask_arr = sub_img.get_fdata() >= 0.1

    >>> mask_img2 = compute_brain_mask(target_img=target_img, implicitmask_arr=implicitmask_arr, verbose=1)
    >>> np.all(mask_img2.get_fdata() == mask_img.get_fdata())
    True
    """

    if target_img is None:
        raise ValueError("target_img is missing")

    if isinstance(target_img, str):
        target_img = nibabel.load(target_img)

    # (1) Tissue mask from priors

    tissue_img = nibabel.load(os.path.join(fsl_home, "data/standard/tissuepriors/avg152T1_%s.img" % tissue_prior))
    tissue_img = resample_to_img(tissue_img, target_img, interpolation='continuous')
    mask_arr = tissue_img.get_fdata().squeeze() >= 0.2
    # mask_arr.sum()

    # (2) Implicit mask if data are provided

    if imgs is not None:
        if isinstance(imgs, list) and len(imgs) >= 1 and isinstance(imgs[0], str):
            imgs_arr, df, target_img_ = img_to_array(imgs)
            assert np.all(target_img.affine == target_img_.affine), "Images do not match target_img"

        elif isinstance(imgs, np.ndarray) and imgs.ndim >= 5:
            imgs_arr = imgs
            assert isinstance(target_img, nibabel.nifti1.Nifti1Image)

        implicitmask_arr = np.ones(imgs_arr.shape[1:], dtype=bool).squeeze()
        if mask_thres_mean is not None:
            implicitmask_arr = implicitmask_arr & (np.abs(np.mean(imgs_arr, axis=0)) >= mask_thres_mean).squeeze()
        if mask_thres_std is not None:
            implicitmask_arr = implicitmask_arr & (np.std(imgs_arr, axis=0) >= mask_thres_std).squeeze()

    # (3) Apply Implicit mask if provided or computed from the data

    if implicitmask_arr is not None:
        assert implicitmask_arr.shape == mask_arr.shape, "Implicit mask shape missmatch"
        mask_arr = implicitmask_arr & mask_arr


    # (4) Remove Brain-Stem and Cerebellum

    if rm_brainstem or rm_cerebellum:
        lobes_img, lobes_labels = fetch_atlas_lobes(fsl_home=fsl_home)
        lobes_img = resample_to_img(lobes_img, target_img, interpolation= 'nearest')
        lobes_arr = lobes_img.get_fdata().astype(int)

        if rm_cerebellum:
            mask_arr[lobes_arr == lobes_labels.index('Cerebellum')] = False

        if rm_brainstem:
            mask_arr[lobes_arr == lobes_labels.index('Brain-Stem')] = False

            # nilearn.image.new_img_like(target_img, mask_arr).to_filename("/tmp/test.nii.gz")

    # (5) Remove small branches
    mask_arr = scipy.ndimage.binary_opening(mask_arr)

    # (6) Avoid isolated clusters: remove all cluster smaller that clust_size_thres
    mask_arr = rm_small_clusters(mask_arr, clust_size_thres=clust_size_thres)

    if verbose >= 1:
        mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
        labels = np.unique(mask_clustlabels_arr)[1:]
        print("Clusters of connected voxels #%i, sizes=" % len(labels),
              [np.sum(mask_clustlabels_arr == lab) for lab in labels])

    return nilearn.image.new_img_like(target_img, mask_arr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('--input', help='list of niftii images', nargs='+', type=str)
    parser.add_argument('--target', help='target niftii image', type=str)
    parser.add_argument('-o', '--output', help='niftii file for the mask', type=str)
    options = parser.parse_args()

    # TODO extends with additional parameters

    if options.target is None:
        parser.print_help()
        raise SystemExit("Error: target is missing.")

    if options.output is None:
        options.output = "mask.nii.gz"

    mask_img = compute_brain_mask(imgs=options.input, target_img=options.target)
    mask_img.to_filename(options.output)
