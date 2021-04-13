#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:15:14 2020

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import mulm
import warnings

class Residualizer:
    """
    Residualization of a Y data on possibly adjusted for other variables.
    Example: Y is a (n, p) array of p-dependant variables, we want to residualize
    for "site" adjusted for "age + sex + diagnosis"

    1) `Residualizer(data=df,
                     formula_res="site",
                     formula_full=site + age + sex + diagnosis")`
    2) `get_design_mat()` will return the numpy (n, k) array design matrix.
    Row selection can be done on both Y and design_mat (Cross-val., etc.)

    3) `fit(Y, design_mat)` fits the model:
    Y = b1 site + b2 age + b3 sex + b4 diagnosis + eps
    => learn and store b1, b2, b3 and b4

    4) `transform(Y, design_mat)` Y and design_mat can contains other
    observations than the ones used in training phase.

    Return Y - b1 site

    Parameters
    ----------
    Y: array (n, p)
        dependant variables

    formula_res: str
        Residualisation formula ex: "site"

    formula_full: str
        Full model (formula) of residualisation containing other variables
        to adjust for. Ex.: "site + age + sex + diagnosis"

    design_mat: array (n, k)
        where Y.shape[0] == design_mat.shape[0] and design_mat.shape[1] is
        the same in fit and transform

    pack_data: boolean (default=False)

    Returns
    -------
    Y: array (n, p)
        Residualized dependant variables

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import scipy.stats as stats
    >>> from nitk.stats import Residualizer
    >>> import seaborn as sns
    >>> np.random.seed(1)
    >>>
    >>> # Dataset with site effect on age
    >>> site = np.array([-1] * 50 + [1] * 50)
    >>> age = np.random.uniform(10, 40, size=100) + 5 * site
    >>> y = -0.1 * age  + site + np.random.normal(size=100)
    >>> data = pd.DataFrame(dict(y=y, age=age, site=site.astype(object)))
    >>>
    >>> # Simple residualization on site
    >>> res_spl = Residualizer(data, formula_res="site")
    >>> yres = res_spl.fit_transform(y[:, None], res_spl.get_design_mat())
    >>>
    >>> # Site residualization adjusted for age
    >>> res_adj = Residualizer(data, formula_res="site", formula_full="age + site")
    >>> yadj = res_adj.fit_transform(y[:, None], res_adj.get_design_mat())
    >>>
    >>> # Site residualization adjusted for age provides higher correlation,
    >>> # and lower stderr than simple residualization
    >>> lm_res = stats.linregress(age, yres.ravel())
    >>> lm_adj = stats.linregress(age, yadj.ravel())
    >>>
    >>> np.allclose((lm_res.slope, lm_res.rvalue, lm_res.stderr),
    >>>             (-0.079187578, -0.623733003, 0.0100242219))
    True
    >>> np.allclose((lm_adj.slope, lm_adj.rvalue, lm_adj.stderr),
    >>>             (-0.110779913, -0.7909219758, 0.00865778640))
    True
    """

    def __init__(self, data, formula_res, formula_full=None):
        warnings.warn(
            "nitk.stats.Residualizer is deprecated, use mulm.residualizer.Residualizer instead. See https://github.com/neurospin/pylearn-mulm",
            DeprecationWarning
        )
        if formula_full is None:
            formula_full = formula_res
        res_terms = mulm.design_matrix(formula=formula_res, data=data)[1].keys()
        self.design_mat, self.t_contrasts, self.f_contrasts = \
            mulm.design_matrix(formula=formula_full, data=data)
        # mask of terms in residualize formula within full model
        self.mask = np.array([cont for term, cont in self.t_contrasts.items()
                              if term in res_terms]).sum(axis=0) == 1

    def get_design_mat(self):
        return self.design_mat

    def fit(self, Y, design_mat):
        """
        Y: array (n, p)
            Dependant variables

        design_mat: array(n, k)
            Design matrix of independant variables
        """

        assert Y.shape[0] == design_mat.shape[0]
        assert self.mask.shape[0] == design_mat.shape[1]
        self.mod_mulm = mulm.MUOLS(Y, design_mat).fit()
        return self

    def transform(self, Y, design_mat=None):

        assert Y.shape[0] == design_mat.shape[0]
        assert self.mask.shape[0] == design_mat.shape[1]
        return Y - np.dot(design_mat[:, self.mask],
                          self.mod_mulm.coef[self.mask, :])

    def fit_transform(self, Y, design_mat):
        self.fit(Y, design_mat)
        return self.transform(Y, design_mat)


class ResidualizerEstimator:
    """Wrap Residualizer into an Estimator compatible with sklearn API.

    Parameters
    ----------
    residualizer: Residualizer
    """

    def __init__(self, residualizer):

        self.residualizer = residualizer
        self.design_mat_ncol = self.residualizer.design_mat.shape[1]

    def fit(self, X, y=None):
        design_mat, Y = self.upack(X)
        return self.residualizer.fit(Y, design_mat)

    def transform(self, X):
        design_mat, Y = self.upack(X)
        return self.residualizer.transform(Y, design_mat)

    def fit_transform(self, X, y=None):
        design_mat, Y = self.upack(X)
        self.residualizer.fit(Y, design_mat)
        return self.residualizer.transform(Y, design_mat)

    def pack(self, Z, X):
        """Pack (concat) Z (design matrix) and X to match scikit-learn pipelines.

        Parameters
        ----------
        Z: array (n, k)
            the design_matrix
        X: array (n, p)
            the input data for scikit-learn: fit(X, y) or transform(X)

        Returns
        -------
        (n, (k+p)) array: [design_matrix, X]
        """
        return np.hstack([Z, X])

    def upack(self, X):
        """Unpack X and Z (design matrix) from X.

        Parameters
        ----------
        X: array (n, (k+p))
            array: [design_matrix, X]

        Returns
        -------
            design_matrix, X
        """
        return X[:, :self.design_mat_ncol], X[:, self.design_mat_ncol:]


def residualize(Y, data, formula_res, formula_full=None):
    """Helper function. See Residualizer
    """
    res = Residualizer(data=data, formula_res=formula_res, formula_full=formula_full)
    return res.fit_transform(Y, res.get_design_mat())


if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from nitk.stats import Residualizer
    import seaborn as sns
    np.random.seed(1)

    # Dataset with site effect on age
    site = np.array([-1] * 50 + [1] * 50)
    age = np.random.uniform(10, 40, size=100) + 5 * site
    y = -0.1 * age + site + np.random.normal(size=100)
    data = pd.DataFrame(dict(y=y, age=age, site=site.astype(object)))

    # Simple residualization on site
    res_spl = Residualizer(data, formula_res="site")
    yres = res_spl.fit_transform(y[:, None], res_spl.get_design_mat())

    # Site residualization adjusted for age
    res_adj = Residualizer(data, formula_res="site", formula_full="age + site")
    yadj = res_adj.fit_transform(y[:, None], res_adj.get_design_mat())

    # Site residualization adjusted for age provides higher correlation, and
    # lower stderr than simple residualization
    lm_res = stats.linregress(age, yres.ravel())
    lm_adj = stats.linregress(age, yadj.ravel())

    np.allclose((lm_res.slope, lm_res.rvalue, lm_res.stderr),
                (-0.079187578, -0.623733003, 0.0100242219))

    np.allclose((lm_adj.slope, lm_adj.rvalue, lm_adj.stderr),
                (-0.110779913, -0.7909219758, 0.00865778640))

    # Plot
    data["yres"] = yres
    data["yadj"] = yadj
    sns.lmplot("age", "y", hue="site", data=data)
    sns.lmplot("age", "yres", hue="site", data=data)
    sns.lmplot("age", "yadj", hue="site", data=data)

    ############################################################################
    # Rezidualizer as pre-processing of supervized prediction:
    # Input: X = age + site + e, target = age
    # Preprocessing:
    # - Residualize X for "site" adjusted for "age"
    # - Learn to predict age on residualized data
    #
    # Since age is used in residualization, it MUST be fitted on training data
    # only.

    # Dataset
    site = np.array([-1] * 50 + [1] * 50)
    age = np.random.uniform(10, 40, size=100) + 5 * site
    X = np.random.randn(100, 5)
    X[:, 0] = -0.1 * age + site + np.random.normal(size=100)
    X[:, 1] = -0.1 * age + site + np.random.normal(size=100)
    demographic_df = pd.DataFrame(dict(age=age, site=site.astype(object)))
    y = age

    # X: input data of the predictive model are the
    # Predictive model
    from sklearn import linear_model
    # from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate
    from sklearn import metrics

    lr = linear_model.Ridge(alpha=1)
    scaler = StandardScaler()
    cv = KFold(n_splits=5, random_state=42)

    ############################################################################
    # Usage 1: Manual slicing of train/test data: use Residualizer

    residualizer = Residualizer(data=demographic_df, formula_res='site',
                                formula_full='site + age')
    Z = residualizer.get_design_mat()
    scores = np.zeros((5, 2))
    for i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[tr_idx, :], X[te_idx, :]
        Z_tr, Z_te = Z[tr_idx, :], Z[te_idx, :]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 1) Fit residualizer
        residualizer.fit(X_tr, Z_tr)

        # 2) Residualize
        X_res_tr = residualizer.transform(X_tr, Z_tr)
        X_res_te = residualizer.transform(X_te, Z_te)

        X_res_tr = scaler.fit_transform(X_res_tr)
        X_res_te = scaler.transform(X_res_te)

        # 3) Fit predictor on train residualized data
        lr.fit(X_res_tr, y_tr)

        # 4) Predict on test residualized data
        y_test_pred = lr.predict(X_res_te)

        # 5) Compute metrics
        scores[i, 0] = metrics.r2_score(y_te, y_test_pred)
        scores[i, 1] = metrics.mean_absolute_error(y_te, y_test_pred)

    scores = pd.DataFrame(scores, columns=['r2', 'mae'])

    print("Mean scores")
    print(scores.mean(axis=0))

    ############################################################################
    # Usage 2: Usage with sklearn pipeline, cross_validate:
    # use ResidualizerEstimator

    residualizer = Residualizer(data=demographic_df, formula_res='site',
                                formula_full='site + age')
    # Extract design matrix and pack it with X
    Z = residualizer.get_design_mat()

    residualizer_wrapper = ResidualizerEstimator(residualizer)
    ZX = residualizer_wrapper.pack(Z, X)

    pipeline = make_pipeline(residualizer_wrapper, StandardScaler(), lr)
    cv_res = cross_validate(estimator=pipeline, X=ZX, y=y, cv=cv, n_jobs=5,
                            scoring=['r2', 'neg_mean_absolute_error'])

    r2 = cv_res['test_r2'].mean()
    mae = np.mean(-cv_res['test_neg_mean_absolute_error'])
    print("CV R2:%.4f, MAE:%.4f" % (r2, mae))
    assert np.allclose(scores.mean(axis=0).values, np.array([r2, mae]))
