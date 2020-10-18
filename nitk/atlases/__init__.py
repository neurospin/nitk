#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:00:32 2020

@author: edouard.duchesnay@cea.fr
"""

from .atlases_loader import fetch_atlas_harvard_oxford, fetch_atlas_cerebellum, fetch_atlas_lobes


__all__ = ['fetch_atlas_harvard_oxford','fetch_atlas_cerebellum', 'fetch_atlas_lobes']