from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def pipeline_behead(pipeline):
    """Separate preprocessing transformers from the prediction head of a pipeline estimator.
    This function assumes that the last step of the pipeline is the prediction head
    (e.g., a classifier or regressor) and all previous steps are preprocessing steps.
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    Returns
    -------
    transformers : Pipeline
        A new Pipeline object containing only the preprocessing steps.
    prediction_head : object
        The last step of the original pipeline, which is the prediction head (e.g., classifier
        Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())])
    >>> transformers, prediction_head = pipeline_behead(pipe)
    >>> # Apply preprocessing to data
    >>> X_preprocessed = transformers.transform(X)
    >>> # Use prediction head on preprocessed data
    >>> prediction_head.predict(X_preprocessed)
    array([0, 0, 1, 1])
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    array([0, 0, 1, 1])
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")

    #pipeline = clone(pipeline)  # Clone the estimator to avoid modifying the original
    preprocessing_steps = pipeline.steps[:-1]
    # Get the last step (the predictor)
    predictor_name, prediction_head = pipeline.steps[-1]
    # Create a new pipeline with only the preprocessing steps
    transformers = Pipeline(preprocessing_steps)
    
    return transformers, prediction_head


    
def get_linear_coefficients(estimator):
    """
    Retrieve the linear coefficient(s) from a scikit-learn estimator or pipeline.

    Handles:
    - Estimator is a Pipeline: returns the coefficient of the last step.
    - Estimator or last step is a GridSearchCV: returns the coefficient of the best_estimator_.
    - If the final estimator has no 'coef_' attribute, returns None.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        A scikit-learn estimator, pipeline, or GridSearchCV object.

    Returns
    -------
    coef : np.ndarray or None
        The linear coefficient(s), or None if not available.

    Examples
    --------

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[0, 1], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> # Direct estimator
    >>> lr = LogisticRegression().fit(X, y)
    >>> get_linear_coefficients(lr).shape
    (1, 2)
    >>> # Pipeline
    >>> pipe = Pipeline([('clf', LogisticRegression())]).fit(X, y)
    >>> get_linear_coefficients(pipe).shape
    (1, 2)
    >>> # GridSearchCV
    >>> param_grid = {'C': [0.1, 1]}
    >>> grid = GridSearchCV(LogisticRegression(), param_grid, cv=2).fit(X, y)
    >>> get_linear_coefficients(grid).shape
    (1, 2)
    >>> # Pipeline + GridSearchCV
    >>> pipe = Pipeline([('clf', LogisticRegression())])
    >>> grid = GridSearchCV(pipe, {'clf__C': [0.1, 1]}, cv=2).fit(X, y)
    >>> get_linear_coefficients(grid).shape
    (1, 2)
    >>> # Non-linear estimator
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> tree = DecisionTreeClassifier().fit(X, y)
    >>> print(get_linear_coefficients(tree))
    None
    """

    while True:

        # Unwrap Pipeline
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]  # get the last step's estimator

        # Unwrap GridSearchCV
        elif isinstance(estimator, GridSearchCV):
            estimator = estimator.best_estimator_

        # Check for coef_ attribute (used by linear models)
        elif hasattr(estimator, 'coef_'):
            return estimator.coef_

        else:
            return None
