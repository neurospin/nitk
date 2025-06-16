from sklearn.model_selection import BaseCrossValidator
import numpy as np

import json

def save_folds(folds, json_file):
    """
    Save the folds of a scikit-learn cross-validation to a JSON file.

    This function takes a list of folds, where each fold is a tuple of train and test indices,
    and saves them to a specified JSON file.

    Parameters:
    -----------
    folds : list of tuples
        A list of tuples where each tuple contains two lists or arrays: the train indices and the test indices.
    json_file : str
        The path to the JSON file where the folds will be saved.

    Example:
    --------
    >>> folds = [(np.array([0, 1, 2]), np.array([3, 4])), (np.array([2, 3, 4]), np.array([0, 1]))]
    >>> save_folds(folds, 'folds.json')
    """
    # Convert numpy arrays to lists for JSON serialization
    folds_list = [(train.tolist(), test.tolist()) for train, test in folds]

    with open(json_file, 'w') as f:
        json.dump(folds_list, f)

def load_folds(json_file):
    """
    Load the folds of a scikit-learn cross-validation from a JSON file.

    This function reads a JSON file containing the folds of a cross-validation and returns them
    as a list of tuples of train and test indices.

    Parameters:
    -----------
    json_file : str
        The path to the JSON file from which the folds will be loaded.

    Returns:
    --------
    list of tuples
        A list of tuples where each tuple contains two lists: the train indices and the test indices.

    Example:
    --------
    >>> folds = load_folds('folds.json')
    >>> print(folds)
    [[[0, 1, 2], [3, 4]], [[2, 3, 4], [0, 1]]]
    """
    with open(json_file, 'r') as f:
        folds_list = json.load(f)

    # Convert lists back to tuples
    folds = [(train, test) for train, test in folds_list]

    return folds


class PredefinedSplit(BaseCrossValidator):
    """
    A custom cross-validator that uses pre-defined train/test splits.

    This class allows you to use pre-defined splits for cross-validation in scikit-learn.
    It is useful when you have specific train/test indices that you want to use directly.

    Parameters:
    -----------
    predefined_splits : list of tuples
        A list of tuples where each tuple contains two arrays: the train indices and the test indices.
        Each tuple represents a single split of the data.
    """

    def __init__(self, predefined_splits=None, json_file=None):
        self.predefined_splits = predefined_splits
        
        if not self.predefined_splits:
            self.predefined_splits = load_folds(json_file)
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features), optional
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into train/test sets.

        Returns:
        --------
        int
            The number of splits, which is the number of pre-defined splits provided.
        """
        return len(self.predefined_splits)

    def split(self, X=None, y=None, groups=None):
        """
        Generates indices to split data into training and test sets.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features), optional
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into train/test sets.

        Yields:
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        for train_idx, test_idx in self.predefined_splits:
            yield train_idx, test_idx

    def to_json(self, json_file):
        save_folds(self.predefined_splits, json_file)

if __name__ == "__main__":

    # Example usage
    # Assume you have a dataset with 10 samples and you have pre-defined splits
    n_samples = 10
    predefined_splits = [
        (np.arange(0, 8), np.arange(8, 10)),  # First fold: samples 0-7 for training, 8-9 for testing
        (np.arange(2, 9), np.arange(0, 2)),  # Second fold: samples 2-8 for training, 0-1 for testing
        # Add more splits as needed
    ]

    # Create an instance of the custom cross-validator
    custom_cv = PredefinedSplit(predefined_splits)

    # Example of using the custom cross-validator with scikit-learn's cross_val_score
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=n_samples, n_features=5, random_state=42)

    # Initialize a classifier
    clf = RandomForestClassifier(random_state=42)

    # Use cross_val_score with the custom cross-validator
    scores = cross_val_score(clf, X, y, cv=custom_cv)

    print("Cross-validation scores:", scores)
    
    
    # Example usage
    # Assume you have a list of folds
    import numpy as np

    folds = [
        (np.array([0, 1, 2]), np.array([3, 4])),
        (np.array([2, 3, 4]), np.array([0, 1]))
    ]

    # Save the folds to a JSON file
    save_folds(folds, 'folds.json')

    # Load the folds from the JSON file
    loaded_folds = load_folds('folds.json')

    print(loaded_folds)
