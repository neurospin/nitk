
def get_residualizer(data, X, residualization_columns, print_log=print):
    """Residualiser

    Parameters
    ----------
    data : DataFrame
        input DataFrame
    X, : numpy Array
    residualization_columns : list of columns
        residualization variables
    print_log : callable, optional
        print function, by default print

    Returns
    -------
    Array, ResidualizerEstimator, str
        Array 
    """
    from mulm.residualizer import Residualizer
    from mulm.residualizer import ResidualizerEstimator
    
    print_log('\n# Residualization')
    # Residualizer
    residualization_formula = "+".join(residualization_columns)
    residualizer = Residualizer(data=data, formula_res=residualization_formula)

    # Extract design matrix and pack it with X
    Z = residualizer.get_design_mat(data=data)
    residualizer_estimator = ResidualizerEstimator(residualizer)
    
    # Repack Z with X
    X = residualizer_estimator.pack(Z, X)

    print_log(residualization_formula)
    print_log("Z.shape:", Z.shape)
    
    return X, residualizer_estimator, residualization_formula


