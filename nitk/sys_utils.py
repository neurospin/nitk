import sys
import os.path
import importlib.util


def import_module_from_path(file_path):

    # Extract the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Use importlib to load the module from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def create_print_log(config):
    """
    Creates and returns a print_log function that logs messages to a file if specified in the config.

    The returned function, `print_log`, will print messages to a specified log file if 'log_filename'
    is present in the config dictionary. Otherwise, it will print messages to the standard output.

    Parameters:
    -----------
    config : dict
        A dictionary containing configuration settings. It should contain a key 'log_filename'
        with the path to the log file if logging to a file is desired.

    Returns:
    --------
    function
        A function that prints messages to the log file or standard output based on the config.

    Example:
    --------
    >>> config = {'log_filename': 'log.txt'}
    >>> print_log = create_print_log(config)
    >>> print_log("This will be written to the log file.")
    """
    def print_log(*args):
        """
        Prints the provided arguments to the log file specified in the config or to the standard output.

        Parameters:
        -----------
        *args : list
            Variable length argument list containing the items to be printed.
        """
        if 'log_filename' in config:
            with open(config['log_filename'], "a") as f:
                print(*args, file=f)
        else:
            print(*args)

    return print_log


if __name__=="__main__":

    # Example usage
    config = {'log_filename': 'log.txt'}
    print_log = create_print_log(config)
    print_log("This will be written to the log file.")

