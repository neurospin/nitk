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
from  nitk.bids import get_keys

def img_to_array(img_filenames, check_same_referential=True, expected=dict()):
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
    >>> from  nitk.image import img_to_array
    >>> import glob
    >>> img_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii")
    >>> imgs_arr, df, ref_img = img_to_array(img_filenames, check_same_referential=True, expected=dict())
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

    df = pd.DataFrame([pd.Series(get_keys(filename)) for filename in img_filenames])


    imgs_nii = [nibabel.load(filename) for filename in df.path]

    ref_img = imgs_nii[0]

    # Check expected dimension
    if 'shape' in expected:
        assert ref_img.get_data().shape == expected['shape']
    if 'zooms' in expected:
        assert ref_img.header.get_zooms() == expected['zooms']

    if check_same_referential: # Check all images have the same transformation
        assert np.all([np.all(img.affine == ref_img.affine) for img in imgs_nii])
        assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in imgs_nii])

    # Load image subjects x chanels (1) x image
    imgs_arr = np.stack([np.expand_dims(img.get_data(), axis=0) for img in imgs_nii])

    return imgs_arr, df, ref_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog=img_to_array.__doc__.split('\n')[1].strip())
    parser.add_argument('--input', help='list of niftii images', nargs='+', type=str)
    parser.add_argument('-o', '--output', help='output prefix for csv file', type=str)
    options = parser.parse_args()

    if options.input is None:
        parser.print_help()
        raise SystemExit("Error: Input is missing.")

    if options.output is None:
        options.output = "imgs"

    imgs_arr, df, ref_img = img_to_array(options.input, check_same_referential=True, expected=dict())

    imgs_arr.tofile(options.output + "_data64.npy")
    df.to_csv(options.output + "_participants.csv", index=False )
    ref_img.to_filename(options.output +  "_imgref.nii.gz")
