#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:37:53 2020

@author: edouard.duchesnay@cea.fr
"""

import re
from collections import OrderedDict

participant_re = re.compile("sub-([^_/]+)")
session_re = re.compile("ses-([^_/]+)/")


def get_keys(filename):
    """
    Extract keys from bids filename. Check consistency of filename.

    Parameters
    ----------
    filename : str
        bids path

    Returns
    -------
    dict
        The minimum returned value is dict(participant_id=<match>,
                             session=<match, '' if empty>,
                             path=filename)

    Raises
    ------
    ValueError
        if match failed or inconsistent match.

    Examples
    --------
    >>> import nitk.bids
    >>> nitk.bids.get_keys('/dirname/sub-ICAAR017/ses-V1/mri/y_sub-ICAAR017_ses-V1_acq-s03_T1w.nii')
    {'participant_id': 'ICAAR017', 'session': 'V1'}
    """
    keys = OrderedDict()

    participant_id = participant_re.findall(filename)
    if len(set(participant_id)) != 1:
        raise ValueError('Found several or no participant id', participant_id, 'in path', filename)
    keys["participant_id"] = participant_id[0]

    session = session_re.findall(filename)
    if len(set(session)) > 1:
        raise ValueError('Found several sessions', session, 'in path', filename)

    elif len(set(session)) == 1:
        keys["session"] = session[0]

    else:
        keys["session"] = ''

    keys["path"] = filename

    return keys
