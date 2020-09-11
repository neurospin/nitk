#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:01:57 2020

@author: ed203246
"""
import os
import numpy as np
import nilearn
import pandas as pd
import nibabel

###############################################################################
# %% Utils

def rm_niftii_extension(filename):
    filename, ext = os.path.splitext(filename)
    if ext == ".gz":
        filename, _ = os.path.splitext(filename)
    return filename


###############################################################################
# %% Lobe atlas from harvard_oxford
###############################################################################

# https://en.wikipedia.org/wiki/Lobes_of_the_brain
lobes_label = {'Background':0, 'Frontal':1, 'Temporal':2, 'Parietal':3, 'Occipital':4, 'Insular':5, 'Limbic':6, 'Brain-Stem':7, 'Cerebellum':8}

output = "."

mapping_dirname = os.path.dirname(os.path.abspath(__file__))
# mapping_dirname = "/home/ed203246/git/nitk/nitk/atlas"

###############################################################################
# Read harvard_oxford

cort = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm", data_dir=None, symmetric_split=False, resume=True, verbose=1)
cort_labels = cort.labels
cort_filename = cort.maps

# FIX bug nilearn.datasets.fetch_atlas_harvard_oxford: Errors in HarvardOxford.tgz / sub-maxprob-thr0-1mm
sub = nilearn.datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm", data_dir='/usr/share/', symmetric_split=False, resume=True, verbose=1)
sub_labels = sub.labels
# sub.maps = os.path.join('/usr/share/data/harvard-oxford-atlases/HarvardOxford', os.path.basename(sub.maps))
sub_filename = sub.maps

cort_img = nibabel.load(cort_filename)
cort_arr  = cort_img.get_fdata().astype(int)
assert len(np.unique(cort_arr)) == len(cort_labels), "Atlas HO : array labels must match labels table"

sub_img = nibabel.load(sub_filename)
sub_arr = sub_img.get_fdata().astype(int)
assert len(np.unique(sub_arr)) == len(sub_labels), "Atlas HO : array labels must match labels table"

assert np.all((cort_img.affine == sub_img.affine))

###############################################################################
# Read harvard_oxford to lobes mapping

cort_ho_to_lobes = pd.read_csv(os.path.join(mapping_dirname, "lobes_HarvardOxford-cort-maxprob-thr0-1mm.csv"))
sub_ho_to_lobes = pd.read_csv(os.path.join(mapping_dirname, "lobes_HarvardOxford-sub-maxprob-thr0-1mm.csv"))

lobes_arr = np.zeros(cort_arr.shape, dtype=int)

###############################################################################
# 1)
cereb_img = nibabel.load("/usr/share/fsl/data/atlases/Cerebellum/Cerebellum-MNIflirt-maxprob-thr0-1mm.nii.gz")
lobes_arr[cereb_img.get_fdata() != 0] = lobes_label['Cerebellum']

###############################################################################
# 2) for each roi in HO atlas apply lobe mapping

# Start with sub
for row in sub_ho_to_lobes.itertuples():
    if not pd.isnull(row.lobe):
        print(row.label, "=>", lobes_label[row.lobe])
        lobes_arr[sub_arr == row.label] = lobes_label[row.lobe]

# Then cort
for row in cort_ho_to_lobes.itertuples():
    if row.name != 'Background':
        print(row.label, "=>", lobes_label[row.lobe])
        lobes_arr[cort_arr == row.label] = lobes_label[row.lobe]


np.unique(lobes_arr)
output_lobes_filename = os.path.join(output, "HarvardOxford_lobes")

pd.DataFrame([[v, k] for k,v in lobes_label.items()], columns=["label", "lobe"]).to_csv(output_lobes_filename + ".csv", index=False)
nibabel.Nifti1Image(lobes_arr, affine=cort_img.affine).to_filename(output_lobes_filename + ".nii.gz")

