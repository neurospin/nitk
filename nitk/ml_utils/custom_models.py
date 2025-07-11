import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Ensure2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # stateless transformer

    def transform(self, X):
        # Case 1: pandas Series (1D) → convert to DataFrame
        if isinstance(X, pd.Series):
            return X.to_frame()

        # Case 2: pandas DataFrame (2D) → return as-is
        if isinstance(X, pd.DataFrame):
            return X

        # Case 3: NumPy array
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

class LogisticRegressionTransformer(LogisticRegression):
    def transform(self, X):
        return Ensure2D().transform(self.decision_function(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class MeanTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Ensure2D().transform(X.mean(axis=1))

    
class GroupFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that groups sets of features and applies a specified transformation to each group.

    This transformer allows for grouping of features and applying transformations such as PCA, mean, LDA,
    logistic regression, or any custom transformer to each group.

    Parameters
    ----------
    groups : dict
        A dictionary where keys are the names of the new features and values are lists of indices or column names
        representing the features to be grouped.
    transformer : str or estimator object
        The transformer to apply to each group. It can be a string like "pca", "mean", "lda", "logistic", "regression",
        or an object that implements fit and transform methods.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> # Sample data with 4 rows and 5 columns
    >>> data = np.array([[1,   2,  3,  4,  5],
                         [6,   7,  8,  9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20]])
    >>> X_df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
    >>> groups = {
    ...     'Group1': [0, 1],  # Grouping first and second features
    ...     'Group2': [2, 3, 4]  # Grouping third, fourth, and fifth features
    ... }
    >>> transformer = GroupFeatureTransformer(groups, transformer="mean")
    >>> X_transformed_df = transformer.fit_transform(X_df)
    >>> print(X_transformed_df)
    [[ 1.5  4. ]
     [ 6.5  9. ]
     [11.5 14. ]
     [16.5 19. ]]
    """
    def __init__(self, groups, transformer):
        self.groups = groups
        self.transformer = transformer

    def _get_transformer(self):
        if isinstance(self.transformer, str):
            if self.transformer == "pca":
                return PCA(n_components=1)
            elif self.transformer == "mean":
                return MeanTransformer()
            elif self.transformer == "lda":
                return LinearDiscriminantAnalysis()
            elif self.transformer == "logistic":
                return LogisticRegression()
            elif self.transformer == "regression":
                return LinearRegression()
            else:
                raise ValueError(f"Unknown transformer string: {self.transformer}")
        elif hasattr(self.transformer, 'fit') and hasattr(self.transformer, 'transform'):
            return self.transformer
        else:
            raise ValueError("transformer must be a string or an object with fit and transform methods")

    def fit(self, X, y=None):

        self.column_transformer_ = ColumnTransformer(
            [(name, self._get_transformer(), features) for name, features in self.groups.items()]
        )
        self.column_transformer_.fit(X, y)
        return self

    def transform(self, X):

        return self.column_transformer_.transform(X)


if __name__ == "__main__":
    
    # %% Example usage for GroupFeatureTransformer
    from sklearn.datasets import load_iris
    data = load_iris()
    mask = np.isin(data.target, [0, 1])
    X_df = pd.DataFrame(data.data, columns=data.feature_names)[mask]
    X, y = data.data[mask], data.target[mask]

    groups = {
        'sepal_features': [0, 1],
        'petal_features': [2, 3]
    }

    # With Numpy arrays
    transformer = GroupFeatureTransformer(groups, transformer="pca")
    transformer.fit_transform(X)

    transformer = GroupFeatureTransformer(groups, transformer="mean")
    transformer.fit_transform(X)

    transformer = GroupFeatureTransformer(groups, transformer="lda")
    transformer.fit_transform(X, y)
    
    transformer = GroupFeatureTransformer(groups, transformer=LogisticRegressionTransformer())
    transformer.fit_transform(X, y)

    # Same with DataFrame
    transformer = GroupFeatureTransformer(groups, transformer="pca")
    transformer.fit_transform(X_df)

    transformer = GroupFeatureTransformer(groups, transformer="mean")
    transformer.fit_transform(X)

    transformer = GroupFeatureTransformer(groups, transformer="lda")
    transformer.fit_transform(X, y)
    
    transformer = GroupFeatureTransformer(groups, transformer=LogisticRegressionTransformer())
    transformer.fit_transform(X, y)