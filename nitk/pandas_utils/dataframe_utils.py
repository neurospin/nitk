import pandas as pd

def expand_key_value_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Expands a column in a DataFrame containing key-value pairs into separate columns.

    The column should contain strings formatted as 'key1-value1_key2-value2_...' 
    where each key-value pair is separated by '_' and keys are separated from values by '-'.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the column in df containing the key-value strings.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the original column replaced by separate columns for each key.

    Example:
    --------
    >>> df = pd.DataFrame([
    ...     ["logistic", "fold-1_size-0.1", 0.5],
    ...     ["logistic", "fold-1_size-0.9", 0.5],
    ...     ["logistic", "fold-2_size-0.1", 0.5],
    ...     ["logistic", "fold-2_size-0.0", 0.5],
    ... ], columns=["model", "params", "auc"])
    >>> expand_key_value_column(df, "params")
       model  fold  size  auc
    0  logistic     1   0.1  0.5
    1  logistic     1   0.9  0.5
    2  logistic     2   0.1  0.5
    3  logistic     2   0.0  0.5
    """
    # Split key-value pairs
    kv_split = df[col].str.split('_').apply(
        lambda items: {k: v for k, v in (item.split('-') for item in items)}
    )
    
    # Convert list of dicts to DataFrame
    kv_df = pd.DataFrame(kv_split.tolist())

    # Convert to appropriate types if possible
    kv_df = kv_df.apply(pd.to_numeric, errors='ignore')

    # Reset index
    df = df.reset_index(drop=True)
    
    # Combine with original DataFrame (excluding the original key-value column)
    df_expanded = pd.concat([df.drop(columns=[col]), kv_df], axis=1)
    
    return df_expanded


def describe_categorical(df):
    """Describes categorical variables in a pandas DataFrame by counting
    occurrences of each category level.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing categorical variables.

    Returns
    -------
    DataFrame
        A DataFrame with counts of each category level for each variable.
    Example
    -------
    Example:
    >>> df = pd.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'bird'],
    ...     'B': ['dog', 'cat', 'fish', 'cat']
    ... })
    >>> print(describe_categorical(df))
         bird  cat  dog  fish
    A      1    2    1     0
    B      0    2    1     1
    """
        # If input is a Series, convert it to a DataFrame with a single column
    if isinstance(df, pd.Series):
        df = df.to_frame('Series')
        
    # Initialize an empty DataFrame to store the result
    result_df = pd.DataFrame()

    # Get all unique categories from all categorical columns
    unique_categories = sorted(df.stack().unique())

    # Iterate over each column in the input DataFrame
    for column in df.columns:
        # Count the occurrences of each category in the current column
        value_counts = df[column].value_counts().reindex(unique_categories, fill_value=0)
        # Add the counts as a new row to the result DataFrame
        result_df[column] = value_counts

    # Transpose the result DataFrame to have variables as rows and categories as columns
    result_df = result_df.T

    return result_df


if __name__=="__main__":
    df = pd.DataFrame([
    ["logistic", "fold-1_size-0.1", 0.5],
    ["logistic", "fold-1_size-0.9", 0.5],
    ["logistic", "fold-2_size-0.1", 0.5],
    ["logistic", "fold-2_size-0.0", 0.5],
    ], columns=["model", "params", "auc"])

    result = expand_key_value_column(df, col="params")
    print(result)