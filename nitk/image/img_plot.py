#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:02 2020

@author: ed203246
"""
import numpy as np
import nilearn
import matplotlib.pyplot as plt


def img_plot_glass_brain(imgs=None, coefs=None, mask_img=None, vmax=None,
                         titles=None, **kwargs):
    """Apply nilearn plot_glass_brain over a list of images, plot in deifferent
    axes.

    Parameters
    ----------
    imgs : [nii], optional
        nii images, if missing, . The default is None.
    coefs : [1D array], optional
        coefficient vectors. The default is None.
    mask_img : nii, optional
        mask image sucha that (mask_img.get_fdata() != 0) == len(coefs).
        The default is None.
    vmax : [float] or float, optional
        Max value. The default is None.
    titles : [str], optional
        Titles. The default is None.

    **kwargs: optionnal arguments for nilearn plot_glass_brain. Example:
        threshold=1e-06, cmap=plt.cm.bwr, plot_abs=False, colorbar=True

    Returns
    -------
    [nii images], fig, axes
    """
    # If no imgs except coef with mask_img
    if imgs is None:
        imgs = list()
        for coef in coefs:
            arr = np.zeros(mask_img.get_fdata().shape)
            arr[mask_img.get_fdata() != 0] = coef
            imgs.append(nilearn.image.new_img_like(mask_img, arr))

    if vmax is None:
        vmax = [np.abs(img.get_fdata()).max() for img in imgs]
    elif np.isscalar(vmax):
        vmax = [vmax for i in range(len(imgs))]

    if titles is None:
        titles = [None for i in range(len(imgs))]

    K = len(imgs)

    fig, axes = plt.subplots(nrows=K, ncols=1, figsize=(11.69, K * 11.69 * .4))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for i, img in enumerate(imgs):
        vmax_ = vmax[i]
        title_ = titles[i]
        # print(i, title_)
        nilearn.plotting.plot_glass_brain(img, vmax=vmax_,
            figure=fig, axes=axes[i], title=title_, **kwargs)

    return imgs, fig, axes
