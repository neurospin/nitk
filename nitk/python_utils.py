from itertools import product

def dict_cartesian_product(*dicts):
    """
    Compute the Cartesian product of multiple dictionaries.

    This function takes multiple dictionaries and returns a new dictionary where each key is a tuple
    representing the Cartesian product of the keys from the input dictionaries, and each value is a
    tuple of the corresponding values from the input dictionaries. If a key/value in the input dictionaries
    are a tuples, they are unpacked in the resulting dictionary.

    Parameters:
    *dicts : dict
        Variable number of dictionaries for which the Cartesian product is to be computed.

    Returns:
    dict
        A dictionary where each key is a tuple of keys from the input dictionaries, and each value
        is a tuple of the corresponding unpacked values.

    Examples:
    >>> dict1 = {("a", "A"): "aA", ("b", "B"): "bB"}
    >>> dict2 = {1: (1, 10), 2: (2, 20)}
    >>> dict_cartesian_product(dict1, dict2)
    {('a', 'A', 1): ('aA', 1, 10),
    ('a', 'A', 2): ('aA', 2, 20),
    ('b', 'B', 1): ('bB', 1, 10),
    ('b', 'B', 2): ('bB', 2, 20)}
    """
    # Compute the Cartesian product of keys
    keys_product = product(*[d.keys() for d in dicts])
    
    # Create the resulting dictionary with unpacked values
    result = {}
    for key_tuple in keys_product:
        
        # Collect values corresponding to the keys in the tuple
        values = []
        keys = []
        for i, key in enumerate(key_tuple):
            
            # flatten values if needed
            value = dicts[i][key]
            if isinstance(value, tuple):
                values.extend(value)
            else:
                values.append(value)
            
            # flatten keys if needed
            if isinstance(key, tuple):
                keys.extend(key)
            else:
                keys.append(key)
                
        result[tuple(keys)] = tuple(values)

    return result
