#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:02 2020

@author: ed203246
"""
import numpy as np
import nilearn
import nilearn.plotting
import matplotlib.pyplot as plt


def plot_glass_brains(imgs=None, thresholds=None, vmax=None,
                      plot_abs=None, colorbars=None, cmaps=None, titles=None, **kwargs):
    """Apply nilearn plot_glass_brain over a list of images, plot in deifferent
    axes.

    Parameters
    ----------
    imgs : [nii], optional
        nii images, if missing, . The default is None.

    vmax : [float] or float, optional
        Max value. The default is None.

    thresholds : [float] or float, optional
        thresholds value. The default is None.

    plot_abs : [bool], optional
        See plot_abs of nilearn.plot_glass_brain

    colorbars : [bool], optional
        See colorbar of nilearn.plot_glass_brain

    cmaps : [plt.cm.*], optional
        Colormap, see cmap of nilearn.plot_glass_brain

    titles : [str], optional
        Titles. The default is None.

    **kwargs: optionnal arguments for nilearn plot_glass_brain. Example:
        threshold=1e-06, cmap=plt.cm.bwr, plot_abs=False, colorbar=True

    Returns
    -------
    [nii images], fig, axes
    """
    if vmax is None:
        vmax = [np.abs(img.get_fdata()).max() for img in imgs]
    elif np.isscalar(vmax):
        vmax = [vmax for i in range(len(imgs))]

    if thresholds is None:
        thresholds = [np.abs(img.get_fdata()).min() for img in imgs]
    elif np.isscalar(thresholds):
        thresholds = [thresholds for i in range(len(imgs))]

    if titles is None:
        titles = [None for i in range(len(imgs))]

    if plot_abs is None:
        plot_abs = [True for i in range(len(imgs))]

    if colorbars is None:
        colorbars = [True for i in range(len(imgs))]

    if cmaps is None:
        cmaps = [None for i in range(len(imgs))]

    K = len(imgs)

    fig, axes = plt.subplots(nrows=K, ncols=1, figsize=(11.69, K * 11.69 * .4))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for i, img in enumerate(imgs):
        vmax_ = vmax[i]
        threshold_ = thresholds[i]
        plot_abs_ = plot_abs[i]
        colorbar_ = colorbars[i]
        cmap_ = cmaps[i]
        title_ = titles[i]
        # print(i, title_)
        nilearn.plotting.plot_glass_brain(img, threshold=threshold_, vmax=vmax_,
            plot_abs=plot_abs_, colorbar=colorbar_, figure=fig, axes=axes[i],
            cmap=cmap_, title=title_, **kwargs)

    return fig, axes

