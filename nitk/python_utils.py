from itertools import product

def dict_cartesian_product(*dicts):
    """
    Compute the Cartesian product of multiple dictionaries.

    This function takes multiple dictionaries and returns a new dictionary where each key is a tuple
    representing the Cartesian product of the keys from the input dictionaries, and each value is a
    tuple of the corresponding values from the input dictionaries. If a value in the input dictionaries
    is a tuple, it is unpacked in the resulting dictionary.

    Parameters:
    *dicts : dict
        Variable number of dictionaries for which the Cartesian product is to be computed.

    Returns:
    dict
        A dictionary where each key is a tuple of keys from the input dictionaries, and each value
        is a tuple of the corresponding unpacked values.

    Examples:
    >>> dict1 = {"a": "A", "b": "B"}
    >>> dict2 = {1: (1, 10), 2: (2, 20)}
    >>> dict_cartesian_product(dict1, dict2)
    {('a', 1): ('A', 1, 10),
     ('a', 2): ('A', 2, 20),
     ('b', 1): ('B', 1, 10),
     ('b', 2): ('B', 2, 20)}
    """
    # Compute the Cartesian product of keys
    keys_product = product(*[d.keys() for d in dicts])

    # Create the resulting dictionary with unpacked values
    result = {}
    for key_tuple in keys_product:
        # Collect values corresponding to the keys in the tuple
        values = []
        for i, key in enumerate(key_tuple):
            value = dicts[i][key]
            if isinstance(value, tuple):
                values.extend(value)
            else:
                values.append(value)

        # Use the flattened values as the value in the result dictionary
        result[key_tuple] = tuple(values)

    return result
