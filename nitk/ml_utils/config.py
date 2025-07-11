import os
import json
import sys
from pathlib import Path


def initialize_config(config_file, config=None):
    """
    Initialize configuration from a JSON file.
    Reads the configuration file, validates required keys, sets the working directory,
    and constructs output paths for logging and results.
    
    returns a dictionary with configuration parameters and sets the working directory.
    required keys in the configuration file include:
    - 'input_data': Path to the input data. can be relative () or absolute.

    Optional keys include:
    - 'WD': Working directory path. () If not provided, it defaults to the directory of the config file.
    - 'target': Name of the target variable column in the input data.
    - 'target_remap': Dictionary for remapping target variable values.
    - 'drop': List of columns to drop from the input data. Can be empty, ie = [].
    - 'residualization': List of columns for residualization, If empty, ie = [], no residualization.
    - 'models_path': Path to the directory containing model definitions.
    - 'metrics': List of metrics to compute.
    - 'LD_LIBRARY_PATH': Optional list of paths to add to the system path.
    
    Output keys include:
    - 'output_models': Trained and serialized models, model predictions, or model summaries
    - 'output_reports': Generated analysis as HTML, PDF, LaTeX, etc.
    - 'prefix': Prefix useful to build output files, derived from the config file name.
    - 'log_filename': Path to the log file.
    - 'cachedir': Path to the directory for caching results.
    
    
    Parameters
    ----------
    config_file : str
        Path to the configuration file (JSON format).
        This file should contain the necessary configuration parameters.
        If the file does not exist, an error will be raised.
    config : dict (optional)
        Dictionary containing configuration parameters.

    Returns
    -------
    config : dict
        Dictionary containing configuration parameters.

    Raises
    ------
    KeyError
        If any required key is missing in the configuration.
        """
    
    # Check if the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # If not provided, load the configuration from the JSON file
    if not config:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
    # Ensure all required keys are present
    required_keys = ['input_data']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key in config: {key}")
    
    # Set Working directory = config file dirname
    if "WD" not in config:
        config["WD"] = os.path.dirname(config_file)
    os.chdir(config["WD"])

    # prefix path for output files: log, cachedir, cv_test
    PREFIX = os.path.splitext(os.path.basename(config_file))[0]
    PREFIX = PREFIX[:-len('_config')] if PREFIX.endswith('_config') else PREFIX
    config["prefix"] = PREFIX

    # Set output paths
    config['log_filename'] = PREFIX + ".log"
    config['cachedir'] = PREFIX + ".cachedir"

    
    # Set output directories
    if 'output_models' not in config:
        config['output_models'] = os.path.join("models")
    Path(config['output_models']).mkdir(parents=True, exist_ok=True)

    if 'output_reports' not in config:
      config['output_reports'] = os.path.join("reports")
    Path(config['output_reports']).mkdir(parents=True, exist_ok=True)
    
    return config
