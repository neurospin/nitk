#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:01:57 2020

@author: ed203246
"""
import os
import os.path
import numpy as np
import nibabel
import nilearn
import nilearn.datasets
from nilearn.image import resample_to_img
import pandas as pd

from  nitk.image import rm_small_clusters

###############################################################################
# Read harvard_oxford
def fetch_atlas_harvard_oxford(data_dir='/usr/share/'):
    """
    Wrap nilearn.datasets.fetch_atlas_harvard_oxford, return both cort and subcort atlases.

    Parameters
    ----------
    data_dir : TYPE, optional
        DESCRIPTION. The default is '/usr/share/'.

    Returns
    -------

    cort_img, cort_labels, sub_img, sub_labels

    """
    cort = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm", data_dir=data_dir,
                                                       symmetric_split=False, resume=True, verbose=1)
    cort_labels = cort.labels
    cort_filename = cort.maps

    # FIX bug nilearn.datasets.fetch_atlas_harvard_oxford: Errors in HarvardOxford.tgz / sub-maxprob-thr0-1mm
    sub = nilearn.datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm", data_dir=data_dir,
                                                      symmetric_split=False, resume=True, verbose=1)
    sub_labels = sub.labels
    # sub.maps = os.path.join('/usr/share/data/harvard-oxford-atlases/HarvardOxford', os.path.basename(sub.maps))
    sub_filename = sub.maps

    cort_img = nibabel.load(cort_filename)
    cort_arr = cort_img.get_fdata().astype(int)
    assert len(np.unique(cort_arr)) == len(cort_labels), "Atlas HO : array labels must match labels table"

    sub_img = nibabel.load(sub_filename)
    sub_arr = sub_img.get_fdata().astype(int)
    assert len(np.unique(sub_arr)) == len(sub_labels), "Atlas HO : array labels must match labels table"
    assert np.all((cort_img.affine == sub_img.affine))

    return cort_img, cort_labels, sub_img, sub_labels


###############################################################################
# Cerebellum
def fetch_atlas_cerebellum(atlas_name="Cerebellum-MNIfnirt-maxprob-thr0-1mm", fsl_home="/usr/share/fsl"):
    """
    Cerebellum atlas

    Parameters
    ----------
    atlas_name : TYPE, optional
        DESCRIPTION. The default is "Cerebellum-MNIfnirt-maxprob-thr0-1mm".
    fsl_home : TYPE, optional
        DESCRIPTION. The default is "/usr/share/fsl".

    Returns
    -------
    cereb_img : nii image
        atlas.
    names : list
        names orderer by label in atlas.
    """
    img_filename = os.path.join(fsl_home, "data/atlases/Cerebellum/%s.nii.gz" % atlas_name)
    xml_file = os.path.join(fsl_home, "data/atlases/Cerebellum_MNIfnirt.xml")

    cereb_img = nibabel.load(img_filename)

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()
    names = [None for i in range(len(list(root.iter('label'))) + 1)]
    names[0] = 'Background'
    for lab in root.iter('label'):
        names[int(lab.attrib['index']) + 1] = lab.text

    # pd.DataFrame(dict(label=np.arange(len(names)), name=names))

    return cereb_img, names


###############################################################################
# Lobes atlas from harvard_oxford
def fetch_atlas_lobes(fsl_home="/usr/share/fsl"):
    """Lobe atlas from harvard_oxford.
    Source: https://en.wikipedia.org/wiki/Lobes_of_the_brain
    lobes_label = {'Background': 0, 'Frontal': 1, 'Temporal': 2, 'Parietal': 3,
                   'Occipital': 4, 'Insular': 5, 'Limbic': 6, 'Brain-Stem': 7,
                   'Cerebellum': 8}

    Parameters
    ----------
    fsl_home : TYPE, optional
        DESCRIPTION. The default is "/usr/share/fsl".

    Returns
    -------
    cereb_img : nii image
        atlas.
    names : list
        names orderer by label in atlas.
    """
    lobes_label = {'Background': 0, 'Frontal': 1, 'Temporal': 2, 'Parietal': 3,
                   'Occipital': 4, 'Insular': 5, 'Limbic': 6, 'Brain-Stem': 7, 'Cerebellum': 8}

    mapping_dirname = os.path.dirname(os.path.abspath(__file__))
    # mapping_dirname = "/home/ed203246/git/nitk/nitk/atlas"

    # Fetch cort, sub and Cerebellum atlases
    cort_img, cort_labels, sub_img, sub_labels = fetch_atlas_harvard_oxford(data_dir=os.path.dirname(fsl_home))
    cereb_img, cereb_names = fetch_atlas_cerebellum(atlas_name="Cerebellum-MNIfnirt-maxprob-thr0-1mm")

    cort_arr = cort_img.get_fdata().astype(int)
    sub_arr = sub_img.get_fdata().astype(int)

    # Read harvard_oxford to lobes mapping
    cort_ho_to_lobes = pd.read_csv(os.path.join(mapping_dirname,
                                                "lobes_HarvardOxford-cort-maxprob-thr0-1mm.csv"))
    sub_ho_to_lobes = pd.read_csv(os.path.join(mapping_dirname,
                                               "lobes_HarvardOxford-sub-maxprob-thr0-1mm.csv"))
    lobes_arr = np.zeros(cort_arr.shape, dtype=int)

    # 1) Cerebellum
    lobes_arr[cereb_img.get_fdata() != 0] = lobes_label['Cerebellum']

    # 2) Sub cortical from HO, apply lobe mapping
    # Start with sub
    for row in sub_ho_to_lobes.itertuples():
        if not pd.isnull(row.lobe):
            # print(row.label, "=>", lobes_label[row.lobe])
            lobes_arr[sub_arr == row.label] = lobes_label[row.lobe]

    # 3) Sub cortical from HO, apply lobe mapping
    for row in cort_ho_to_lobes.itertuples():
        if row.name != 'Background':
            # print(row.label, "=>", lobes_label[row.lobe])
            lobes_arr[cort_arr == row.label] = lobes_label[row.lobe]

    reverse = {v: k for k, v in lobes_label.items()}
    lab = [k for k in reverse]
    lab.sort()
    names = [reverse[k] for k in lab]
    img = nibabel.Nifti1Image(lobes_arr, affine=cort_img.affine)

    return img, names

def fetch_cortex(fsl_home = "/usr/share/fsl"):
    """Cortex at 1mm iso in MNI.

    Cortex = (cortex atlas_harvard_oxford) & (GM tissue > 0.4) & brain mask + remove isolated
    Returns
    -------

    """
    MNI152_T1_1mm_brain_mask_img = nibabel.load(os.path.join(fsl_home, "data/standard/MNI152_T1_1mm_brain_mask.nii.gz"))

    # 1) atlas_harvard_oxford atlas
    cort_img, cort_labels, sub_img, sub_labels = fetch_atlas_harvard_oxford()
    # cort_labels = {name:lab for lab, name in enumerate(cort_labels)}
    # sub_labels = {name:lab for lab, name in enumerate(sub_labels)}

    # 2) Tissue priors
    tissue_prior = "gray"
    gm_prior_img = nibabel.load(os.path.join(fsl_home, "data/standard/tissuepriors/avg152T1_%s.img" % tissue_prior))
    gm_prior_img = resample_to_img(gm_prior_img, MNI152_T1_1mm_brain_mask_img, interpolation='continuous')
    # Intersect with cortex
    assert np.all(gm_prior_img.affine == cort_img.affine)
    cort_arr = (gm_prior_img.get_fdata().squeeze() >= 0.4) & \
               (cort_img.get_fdata() != 0) & \
               (MNI152_T1_1mm_brain_mask_img.get_fdata() != 0)

    # 3) Avoid isolated clusters: remove all cluster smaller that clust_size_thres
    cort_arr = rm_small_clusters(cort_arr, clust_size_thres=10000)
    return nilearn.image.new_img_like(cort_img, cort_arr)


###############################################################################
# Neuromorphometrics

def match_atlas_neuromorphometrics_roi_to_lobe():
    """Roi to lobe mapping of Neuromorphometrics atlas.
    
    Returns
    -------
    map_roi_to_lobes : DataFrame
        atlas.

    Notes
    -----
    Mapping done by iris menu <iris.menu@hotmail.fr> 2021/07/08

    lobes_Neuromorphometrics.csv : ROI to lobe mapping
    http://neuromorphometrics.com/ParcellationProtocol_2010-04-05.PDF
    https://scalablebrainatlas.incf.org/human/NMM1103
    """
    mapping_dirname = os.path.dirname(os.path.abspath(__file__))
    map_roi_to_lobes = pd.read_csv(os.path.join(mapping_dirname,
                                                "lobes_Neuromorphometrics.csv"))
    return map_roi_to_lobes


if __name__ == "__main__":
    # Set default values to parameters
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('atlas', help='atlas name in harvard_oxford, cerebellum, lobes', type=str)
    parser.add_argument('-o', '--output',
        help='Output directory', type=str, default=".")

    if options.atlas == 'harvard_oxford':
        cort_img, cort_labels, sub_img, sub_labels = fetch_atlas_harvard_oxford()
        cort_img.to_filename(os.path.join(options.output, "%s-cort.nii.gz" % options.atlas))
        sub_img.to_filename(os.path.join(options.output, "%s-sub.nii.gz" % options.atlas))
        cort_df = pd.DataFrame(dict(name=cort_labels, label=np.arange(len(cort_labels))))
        sub_df = pd.DataFrame(dict(name=sub_labels, label=np.arange(len(sub_labels))))
        cort_df.to_filename(os.path.join(options.output, "%s-cort.csv" % options.atlas), index=False)
        sub_df.to_filename(os.path.join(options.output, "%s-sub.csv" % options.atlas), index=False)

    elif options.atlas == 'cerebellum':
        cereb_img, names = fetch_atlas_cerebellum()
        cereb_img.to_filename(os.path.join(options.output, "%s.nii.gz" % options.atlas))
        df = pd.DataFrame(dict(name=names, label=np.arange(len(names))))
        df.to_filename(os.path.join(options.output, "%s.csv" % options.atlas), index=False)

    elif options.atlas == 'lobes':
        img, names = fetch_atlas_lobes()
        img.to_filename(os.path.join(options.output, "%s.nii.gz" % options.atlas))
        df = pd.DataFrame(dict(name=names, label=np.arange(len(names))))
        df.to_filename(os.path.join(options.output, "%s.csv" % options.atlas), index=False)
