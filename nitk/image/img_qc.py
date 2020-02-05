#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:27:15 2020

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import pandas as pd
import nibabel
import argparse
from  nitk.bids import get_keys

def img_qc(imgs, participant_id, imgs_arr=None):
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

from  nitk.image import img_to_array, compute_brain_mask
import glob
imgs = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii")
imgs_arr, df, ref_img = img_to_array(imgs)
mask_img = compute_brain_mask(imgs, ref_img)

mask_arr = mask_img.get_data() > 0
participant_id = df.participant_id


if mask_arr is not None:
    X = imgs_arr.squeeze()[:, mask_arr]
else: # Flatten all but the first axis
    X = imgs_arr.reshape(imgs_arr.shape[0], -1)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(42)
%matplotlib qt


###############################################################################
# PCA

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
PC = pca.transform(X)
df = pd.DataFrame(PC, columns=["PC1","PC2"], index=participant_id)
df["participant_id"] = df.index


# New figure make it very large to see subject labels
fig, ax = plt.subplots(figsize=(30, 20))

plt.scatter(df['PC1'], df['PC2'])
for i in range(0,df.shape[0]):
     ax.text(df.PC1[i]+0.01, df.PC2[i],
     df.participant_id[i], horizontalalignment='left',
     size='medium', color='black', weight='semibold')

#plt.text(df['PC1'], df['PC1'], [v for v in participant_id.values], fontsize=9)
plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
plt.axis('equal')
plt.tight_layout()

###############################################################################
# Corr mat

# noiseidx = np.random.choice(np.arange(X.shape[0]), 50)
# X[noiseidx, :] = X[noiseidx, :] + np.random.randn(50, X.shape[1]) / 5
corr = np.corrcoef(X)
F = 0.5 * np.log((1. + corr) / (1. - corr))
np.fill_diagonal(F, 1)

#F = F[:3, :][:, :3]
#F = np.array([[1, 0.8, 0.1], [0.8, 1, 0.9], [0.1, 0.9, 1]])

corr_mean = (F.sum(axis=1) - 1) / (F.shape[0] - 1)
corr_order = corr_mean.argsort()

Freorder =  F[np.ix_(corr_order, corr_order)]
assert np.allclose((Freorder.sum(axis=1) - 1) / (Freorder.shape[0] - 1), corr_mean[corr_order])

pd.DataFrame(Freorder, )
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.color_palette("RdBu_r", 110)
# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(Freorder, mask=None, cmap=cmap, vmin=-1, vmax=1, center=0)

cor = pd.DataFrame(dict(participant_id=participant_id[corr_order], corr_mean=corr_mean[corr_order]))
