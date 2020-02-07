#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:15:14 2020

@author: ed203246

res = Residualizer(data, formula_res, formula_full=None)
design_mat = res.get_design_mat()
mod = ???

for tr, te in cv.split(X, y):
    Xtr ytr, design_mat_tr = X[tr, :], y[tr], design_mat[tr,: ]
    Xte yte, design_mat_te = X[te, :], y[te], design_mat[te,: ]
    Xtr = res.fit_transform(Xtr, design_mat_tr)
    mod.fit(Xtr, ytr)
    Xte = res.transform(Xte, design_mat_te)
    mod.predict(Xte)



"""
import numpy as np
import mulm

class Residualizer:
    def __init__(self, data, formula_res, formula_full=None):
        if formula_full is None:
            formula_full = formula_res
        res_terms = mulm.design_matrix(formula=formula_res, data=data)[1].keys()
        self.design_mat, self.t_contrasts, self.f_contrasts = mulm.design_matrix(formula=formula_full, data=data)
        # mask of terms in residualize formula within full model
        self.mask = np.array([cont  for term, cont in self.t_contrasts.items() if term in res_terms]).sum(axis=0) == 1

    def get_design_mat(self):
        return self.design_mat

    def fit(self, Y, design_mat):
        self.mod_mulm = mulm.MUOLS(Y, design_mat).fit()
        return self

    def transform(self, Y):
        return Y - np.dot(self.design_mat[:, self.mask], self.mod_mulm.coef[self.mask, :])

    def fit_transform(self, Y, design_mat):
        self.fit(Y, design_mat)
        return self.transform(Y)

def oldies_residualize(Y, formula_res, data, formula_full=None):
    """
    Residualisation of adjusted residualization.

    Parameters
    ----------
    Y: array (n, p), dependant variables
    formula_res: str, residualisation formula ex: "site":
    1) Fit  Y = b0 + b1 site + eps
    2) Return Y - b0 - b1 site
    data: DataFrame of independant variables
    formula_full:  str, full model formula (default None) ex: "age + sex + site + diagnosis". If not Null residualize
    performs an adjusted residualization:
    1) Fit Y = b1 age + b2 sex + b3 site + b4 diagnosis + eps
    2) Return Y - b3 site

    Returns
    -------
    Y: array (n, p), of residualized dependant variables
    """
    if formula_full is None:
        formula_full = formula_res

    res_terms = mulm.design_matrix(formula=formula_res, data=data)[1].keys()

    X, t_contrasts, f_contrasts = mulm.design_matrix(formula=formula_full, data=data)

    # Fit full model
    mod_mulm = mulm.MUOLS(Y, X).fit()

    # mask of terms in residualize formula within full model
    mask = np.array([cont  for term, cont in t_contrasts.items() if term in res_terms]).sum(axis=0) == 1

    return Y -  np.dot(X[:, mask], mod_mulm.coef[mask, :])


if __name__ == '__main__':

    import pandas as pd
    import seaborn as sns

    # Dataset with site effect on age
    sex = np.random.choice([0, 1], 100)
    site = np.array([-1] * 50 + [1] * 50)
    age = np.random.uniform(10, 40, size=100) + 10 * site
    y = -0.1 * age + 0.0 * sex + site + np.random.normal(size=100)
    data = pd.DataFrame(dict(y=y, age=age, sex=[str(s) for s in sex], site=[str(s) for s in sex]))

    res_spl = Residualizer(data, formula_res="site")
    data["yres"] = res_spl.fit_transform(y[:, None], res_spl.get_design_mat())

    res_adj = Residualizer(data, formula_res="site", formula_full="age + sex + site")
    data["yadj"] = res_adj.fit_transform(y[:, None], res_adj.get_design_mat())

    # Requires adjusted residualization
    sns.lmplot("age", "y", hue="site", data=data)
    sns.lmplot("age", "yres", hue="site", data=data)
    sns.lmplot("age", "yadj", hue="site", data=data)

