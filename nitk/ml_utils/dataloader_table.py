import pandas as pd
from nitk.pandas_utils.dataframe_utils import describe_categorical

def get_y(data, target_column, remap_dict=None, print_log=print):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    target_column : _type_
        _description_
    remap_dict: dict
        remap target, defualts None
    print_log : _type_, optional
        _description_, by default print

    Returns
    -------
    _type_
        _description_

    Yields
    ------
    _type_
        _description_
    """
    print_log('\n# y (target)\n"%s", counts:' % target_column)
    print_log(describe_categorical(data[target_column]))

    if remap_dict:
        y = data[target_column].map(remap_dict)
        print_log('After remapping, counts:')
        print_log(describe_categorical(y))
    else:
        y = data[target_column]
    return y


def get_X(data, input_columns, print_log=print):
    """Get input Data. Perform dummy codings for categorical variables

    Parameters
    ----------
    data : DataFrame
        Input dataFrame
    input_columns : list
        input columns
    print_log: callable function, default print

    Returns
    -------
        pd.DataFrame: input Data
    """
    X = data[input_columns]
    ncol = X.shape[1]

    print_log('\n# X (Input data)')
    print_log(X.describe(include='all').T)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print_log("\nCategorical columns:", categorical_cols)
        for v in categorical_cols:
            print_log(v, describe_categorical(data[v]))
        X = pd.get_dummies(X, dtype=int)
        print_log('\nAfter coding')
        print_log(X.describe(include='all').T)
        print_log('%i dummies variable created' %  (X.shape[1] - ncol))

    return X
