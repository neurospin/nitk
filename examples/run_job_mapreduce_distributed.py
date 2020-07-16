#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:24:44 2020

@author: edouard.duchesnay@cea.fr
"""

import os
import numpy as np
import pickle
import time
from pathlib import Path
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold

from nitk.utils import dict_product, MapReduce, reduce_cv_classif


###############################################################################
# Working directory

WD = os.path.join(str(Path.home()), "_nitk_job_mapreduce_sklearn")
os.makedirs(WD, exist_ok=True)

xls_filename =  os.path.join(WD, "classif_scores.xlsx")
models_filename =  os.path.join(WD, "classif_models.pkl")

###############################################################################
# Dataset

X, y = make_classification(n_samples=100,
    n_features=1000, n_informative=50, n_redundant=0, n_classes=2,
    random_state=42)

###############################################################################
# function to be applied by the mapper

def fit_predict(key, estimator, split):
    print(key)
    start_time = time.time()
    train, test = split
    X_train, X_test,  y_train = X[train, :], X[test, :], y[train]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optional this step lookup for model already fitted in KEY_VALS
    try: # if coeficient can be retrieved given the key
        estimator.coef_ = KEY_VALS[key]['coef']
    except: # if not fit
        estimator.fit(X_train, y_train)

    y_test = estimator.predict(X_test)
    score_test = estimator.decision_function(X_test)
    # score_train = estimator.decision_function(X_train)
    try:
        coef = estimator.coef_
    except:
        coef = None
    time_elapsed = round(time.time() - start_time, 2)

    return dict(y_test=y_test, score_test=score_test, time=time_elapsed,
                coef=coef)


###############################################################################
# Configure input key/value pairs = models x CV folds in a dictionary

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
cv_dict = {"CV%i" % fold:split for fold, split in enumerate(cv.split(X, y))}
# cv_dict["ALL"] = [np.arange(X.shape[0]), np.arange(X.shape[0])]

Cs = np.logspace(-3, 3, 7)
l2 = {"l2_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}

estimators_dict = dict()
estimators_dict.update(l2)
key_values_input = dict_product(estimators_dict, cv_dict)
print("Nb Tasks=%i" % len(key_values_input))


###############################################################################
# Centralized Mapper => output key/value pairs

key_vals_output = MapReduce(n_jobs=5, pass_key=True, verbose=20).map(fit_predict, key_values_input)

# Reducer => CV scores

cv_scores = reduce_cv_classif(key_vals_output, cv_dict, y_true=y)
cv_scores_mean = cv_scores.groupby(["param_0"]).mean().reset_index()

with open(models_filename, 'wb') as fd:
    pickle.dump(key_vals_output, fd)

with pd.ExcelWriter(xls_filename) as writer:
    cv_scores.to_excel(writer, sheet_name='folds', index=False)
    cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)


###############################################################################
# Distributed Mapper  => output key/value pairs

# create a shared directory to store the results
shared_dir = os.path.join(WD, "mapreduce")
os.makedirs(shared_dir, exist_ok=True)


mp = MapReduce(n_jobs=5, shared_dir=shared_dir, pass_key=True, verbose=20)
mp.map(fit_predict, key_values_input)
key_vals_output = mp.reduce_collect_outputs()

self=mp
if cv_scores is not None:
    print("All task completed results could be loaded")
    print(cv_scores)
    assert res_multi == res_single
else:
    print("Some tasks were not finished aborted redcue")


