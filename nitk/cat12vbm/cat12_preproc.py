import glob
import argparse
import sys; sys.path.extend(['../', '../..', '.'])
from nitk.bids.bids_utils import get_keys


def get_missing_files(base_files, cat12_files):
    all_mappings = {get_unique_key(get_keys(file)): file for file in base_files}
    all_keys = all_mappings.keys()
    cat12_keys = [get_unique_key(get_keys(file)) for file in cat12_files]
    missing_keys = list(set(all_keys) - set(cat12_keys))
    missing_path = [all_mappings[k] for k in missing_keys]
    return missing_path

def get_unique_key(keys):
    return '$'.join([keys['participant_id'], keys['session']])


def retrieve_path(key):
    return key.split('$')[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input MRI .nii files given to CAT 12', nargs='+', type=str)
    parser.add_argument('-r', '--report', help='Output report/*.pdf files given by CAT 12 at the end', nargs='+', type=str)

    options = parser.parse_args()
    missing_files = get_missing_files(options.input, options.report)
    print('##Regex:\n')
    regex = '.*sub-({}).*'.format('|'.join([get_keys(f)['participant_id'] for f in missing_files]))
    print(regex)
    print('##List files:\n')
    print('\n'.join(missing_files))
