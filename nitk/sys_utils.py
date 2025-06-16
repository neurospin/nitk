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