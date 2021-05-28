#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:12:53 2020

@author: edouard.duchesnay@cea.fr
"""

from .img_to_array import img_to_array  # DEPRECATED
from .img_to_array import niimgs_to_array, niimgs_bids_to_array, arr_to_4dniimg
from .img_to_array import vec_to_niimg
from .img_search_light import search_light

from .img_to_array import flat_to_array
from .img_brain_mask import compute_brain_mask, rm_small_clusters
from .img_global_operations import global_scaling, center_by_site
from .img_plot import plot_glass_brains
from .img_shapes import make_sphere
#from .img_array_to_img import vec_to_niimg

__all__ = ['img_to_array',  # DEPRECATED
           'niimgs_to_array',
           'arr_to_4dniimg',
           'vec_to_img',
           'flat_to_array'
           'compute_brain_mask', 'rm_small_clusters',
           'global_scaling',
           'center_by_site',
           'plot_glass_brains',
           'make_sphere']
