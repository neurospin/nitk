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
import sklearn.linear_model as lm
from sklearn.model_selection import StratifiedKFold

from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif

###############################################################################
# Working directory, and results filenames

WD = os.path.join(str(Path.home()), "_nitk_job_mapreduce_sklearn")
os.makedirs(WD, exist_ok=True)

xls_filename =  os.path.join(WD, "classif_scores.xlsx")
models_filename =  os.path.join(WD, "classif_models.pkl")

###############################################################################
# Dataset

X, y = make_classification(n_samples=500, class_sep=1.0,
    n_features=300000, n_informative=5, n_redundant=1000, n_classes=2,
    random_state=42)

###############################################################################
# 1) Mapper function

def fit_predict(key, estimator, split):
    # print(key)
    # to debug, pick a key and corresponding values
    # key = list(key_values_input.keys())[0]
    # estimator, split = key_values_input[key]

    start_time = time.time()
    train, test = split
    X_train, X_test,  y_train = X[train, :], X[test, :], y[train]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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


print("""
###############################################################################
# 2) Input key/value pairs = models x CV folds in a dictionary""")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
cv_dict = {"CV%02d" % fold:split for fold, split in enumerate(cv.split(X, y))}
# cv_dict["ALL"] = [np.arange(X.shape[0]), np.arange(X.shape[0])]

#Cs = np.logspace(-3, 3, 7)
#Cs = np.logspace(-1, 1, 3)
Cs = np.logspace(-1, 0, 2)

l2 = {"l2_C:%.6f" % C: lm.LogisticRegression(C=C, class_weight='balanced', fit_intercept=False) for C in Cs}

estimators_dict = dict()
estimators_dict.update(l2)
key_values_input = dict_product(estimators_dict, cv_dict)
print("# Nb Tasks=%i" % len(key_values_input))


print("""
###############################################################################
# 3.1) Centralized Mapper: input key/value pairs => output key/value pairs""")

start_time = time.time()
key_vals_output = MapReduce(n_jobs=5, pass_key=True, verbose=20).map(fit_predict, key_values_input)
print("# Centralized mapper completed in %.2f sec" % (time.time() - start_time))
cv_scores = reduce_cv_classif(key_vals_output, cv_dict, y_true=y)
cv_scores_mean = cv_scores.groupby(["param_0"]).mean().reset_index()
print(cv_scores_mean)

with open(models_filename, 'wb') as fd:
    pickle.dump(key_vals_output, fd)

with pd.ExcelWriter(xls_filename) as writer:
    cv_scores.to_excel(writer, sheet_name='folds', index=False)
    cv_scores_mean.to_excel(writer, sheet_name='mean', index=False)


print("""
###############################################################################
# 3.2) Distributed Mapper: input key/value pairs => output key/value pairs""")

# Create a shared directory to store the results
shared_dir = os.path.join(WD, "mapreduce")

if os.path.exists(shared_dir):
    print("# Existing shared dir, delete for fresh restart: ")
    print("rm -rf %s" % shared_dir)

os.makedirs(shared_dir, exist_ok=True)

start_time = time.time()
mp = MapReduce(n_jobs=5, shared_dir=shared_dir, pass_key=True, verbose=20)
mp.map(fit_predict, key_values_input)
key_vals_dist_output = mp.reduce_collect_outputs()


print("""
###############################################################################
# 4) Reducer: output key/value pairs => CV scores""")

if key_vals_dist_output is not None:
    print("# Distributed mapper completed in %.2f sec" % (time.time() - start_time))

    cv_scores_dist = reduce_cv_classif(key_vals_dist_output, cv_dict, y_true=y)
    cv_scores_dist_mean = cv_scores.groupby(["param_0"]).mean().reset_index()


    ###########################################################################
    # Check distributed mapper == Centralized

    cv_scores.sort_values(["param_0", "fold"], inplace=True, ignore_index=True)
    cv_scores_dist.sort_values(["param_0", "fold"], inplace=True, ignore_index=True)
    assert np.all(cv_scores == cv_scores_dist), "Centralized mapper != distributed mapper"

else:
    print("# Some tasks were not completed, skip reduce")


