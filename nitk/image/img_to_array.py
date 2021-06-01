#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:45:42 2020

@author: edouard.duchesnay@cea.fr

Load images assuming paths contain a BIDS pattern to retrieve participant_id such sub-<participant_id>


"""

import numpy as np
import pandas as pd
import nibabel
import argparse
from nitk.bids import get_keys
import nilearn

def niimgs_to_array(niimgs):
    """Ni images to array

    Parameters
    ----------
    niimgs : [nifti images] or single 4D nitfi image.

    Returns
    -------
    imgs_arr : array
        (n_subjects, 1, , image_axis0, image_axis1, ...).

    Examples
    --------
    >>> import numpy as np
    >>> import nilearn
    >>> from nilearn import datasets, image
    >>> rsn = datasets.fetch_atlas_smith_2009()['rsn10']
    >>> niimgs_list = [image.index_img(rsn, 0), image.index_img(rsn, 1)]
    >>> from nitk.image import niimgs_to_array
    >>> # From list of ni images
    >>> arr1 = niimgs_to_array(niimgs_list)
    >>> arr1.shape
    (2, 1, 91, 109, 91)
    >>> # From 4D ni images
    >>> arr2 = niimgs_to_array(image.concat_imgs(niimgs_list))
    >>> arr2.shape
    (2, 1, 91, 109, 91)
    >>> np.all(arr1 == arr2)
    True
    """
    if  isinstance(niimgs, list):
        return np.stack([np.expand_dims(img.get_fdata(), axis=0) for img in niimgs])

    elif isinstance(niimgs, nibabel.nifti1.Nifti1Image) and niimgs.ndim == 4:
        arr = niimgs.get_fdata()
        arr = np.expand_dims(arr, axis=0)
        arr = np.moveaxis(arr, -1, 0)
        return arr

    else:
        assert False, "Wrong input"

def array_to_niimgs(ref_niimg, arr):
    """Arr to 4d nii images.

    Parameters
    ----------
    ref_niimg : niimg
        reference image.
    arr : array (n_subjects, 1, , image_axis0, image_axis1, ...).
        data array.
    Returns
    -------
    4D nitfi images.

    Examples
    --------
    >>> import numpy as np
    >>> import nilearn
    >>> from nilearn import datasets, image
    >>> rsn = datasets.fetch_atlas_smith_2009()['rsn10']
    >>> niimgs_list = [image.index_img(rsn, 0), image.index_img(rsn, 1)]
    >>> # Build 4D with nilearn
    >>> niimgs_4d = nilearn.image.concat_imgs(niimgs_list)
    >>> from nitk.image import niimgs_to_array, array_to_niimgs
    >>> # Build arr and convert back to 4D
    >>> arr = niimgs_to_array(niimgs_list)
    >>> niimgs_4d_ = array_to_niimgs(ref_niimg=niimgs_list[0], arr=arr)
    >>> np.all(niimgs_4d.get_fdata() == niimgs_4d_.get_fdata())
    True
    """

    arr4d_ = np.moveaxis(arr.squeeze(), 0, -1)
    return nilearn.image.new_img_like(ref_niimg, arr4d_)


def niimgs_to_arr(niimgs):
    """4d nii images to arr.


    Parameters
    ----------
    niimgs : 4D nitfi images.
        4D nitfi images..

    Returns
    -------
    arr.

    """

    return np.moveaxis(niimgs.get_fdata().squeeze(), 0, -1)

def vec_to_niimg(vec, mask_img):
    """Flat vector to nii image, where values within mask are set to vec.


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
    assert mask_arr.sum() == len(vec), "Missmatch between mask and flat vector"
    val_arr = np.zeros(mask_img.shape)
    val_arr[mask_arr] = vec
    return nibabel.Nifti1Image(val_arr, affine=mask_img.affine)


def flat_to_array(data_flat, mask_arr, fill=0):
    """Flat data (n_subjects x n_features withing a mask) to n_subjects x mask_arr.shape.


    Parameters
    ----------
    data_flat : array
        (n_subjects x n_features withing a mask).
    mask_arr : array
        3D mask.
    fill : float
        out-off mask filling data
    Returns
    -------
    array (n_subjects x mask_arr.shape)

    Examples
    --------
    >>> import numpy as np
    >>> from nitk.image import flat_to_array
    >>> data_flat = np.zeros((2, 9))
    >>> data_flat[0, :] = np.arange(0, 9, 1)
    >>> data_flat[1, :] = np.arange(0, 90, 10)
    >>> mask_arr = np.ones((3, 3), dtype=bool)
    >>> np.all(flat_to_array(data_flat, mask_arr).squeeze()[:, mask_arr] == data_flat)
    True
    """
    n_subjects = data_flat.shape[0]
    assert mask_arr.sum() == data_flat.shape[1]
    arr = np.zeros([n_subjects, 1] + list(mask_arr.shape))
    arr[::] = fill
    arr[:, 0, mask_arr] = data_flat

    return arr


def niimgs_bids_to_array(img_filenames, check_same_referential=True, expected=dict()):
    """
    Convert nii images to array (n_subjects, 1, , image_axis0, image_axis1, ...)
    Assume BIDS organisation of file to retrive participant_id and session.

    Parameters
    ----------
    img_filenames : [str]
        path to images

    check_same_referential : bool
        if True (default) check that all image have the same referential.

    expected : dict
        optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
        imgs_arr : array (n_subjects, 1, , image_axis0, image_axis1, ...)
            The array data structure (n_subjects, n_channels, image_axis0, image_axis1, ...)

        df : DataFrame
            With column: 'participant_id', 'session', 'path'

        ref_img : nii image
            The first image used to store referential and all information relative to the images.

    Example
    -------
    >>> from  nitk.image import niimgs_bids_to_array
    >>> import glob
    >>> img_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii")
    >>> imgs_arr, df, ref_img = niimgs_bids_to_array(img_filenames)
    >>> print(imgs_arr.shape)
    (171, 1, 121, 145, 121)
    >>> print(df.shape)
    (171, 3)
    >>> print(df.head())
      participant_id session                                               path
    0       ICAAR017      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    3  STARTLB160534      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    4       ICAAR048      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...

    """
    try:
        df = pd.DataFrame([pd.Series(get_keys(filename)) for filename in img_filenames])
    except:
        df = pd.DataFrame(dict(path=img_filenames))

    imgs_nii = [nibabel.load(filename) for filename in df.path]

    ref_img = imgs_nii[0]

    # Check expected dimension
    if 'shape' in expected:
        assert ref_img.get_fdata().shape == expected['shape']
    if 'zooms' in expected:
        assert ref_img.header.get_zooms() == expected['zooms']

    if check_same_referential: # Check all images have the same transformation
        assert np.all([np.all(img.affine == ref_img.affine) for img in imgs_nii])
        assert np.all([np.all(img.get_fdata().shape == ref_img.get_fdata().shape) for img in imgs_nii])

    # Load image subjects x channels (1) x image
    #imgs_arr = np.stack([np.expand_dims(img.get_fdata(), axis=0) for img in imgs_nii])
    imgs_arr = niimgs_to_array(imgs_nii)

    return imgs_arr, df, ref_img


img_to_array = niimgs_bids_to_array # DEPRECATED



if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog=niimgs_bids_to_array.__doc__.split('\n')[1].strip())
    parser.add_argument('--input', help='list of niftii images', nargs='+', required=True, type=str)
    parser.add_argument('-o', '--output', help='output prefix for csv file', type=str)
    options = parser.parse_args()

    if options.input is None:
        parser.print_help()
        raise SystemExit("Error: Input is missing.")

    if options.output is None:
        options.output = "imgs"

    imgs_arr, df, ref_img = niimgs_bids_to_array(options.input, check_same_referential=True, expected=dict())

    imgs_arr.tofile(options.output + "_data64.npy")
    df.to_csv(options.output + "_participants.csv", index=False )
    ref_img.to_filename(options.output +  "_imgref.nii.gz")
