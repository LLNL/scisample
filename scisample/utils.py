"""
Helper functions for ``scisample``.
"""

import yaml
import logging

LOG = logging.getLogger(__name__)


class SamplingError(Exception):
    """Base class for exceptions in this module."""
    pass


def log_and_raise_exception(msg):
    LOG.error(msg)
    raise(SamplingError(msg))


def read_yaml(filename):
    """
    Read a yaml file; return its contents as a dictionary.

    :param filename: Name of file to read.
    :returns: Dictionary of file contents.
    """
    with open(filename, 'r') as _file:
        content = yaml.safe_load(_file)
    return content
    

def read_csv(filename):
    """
    Reads csv files and returns them as a list of lists.
    """
    with open(filename, 'r') as _file:
        content = _file.readlines()
    return [line.strip().split(',') for line in content]


def transpose_tabular(rows):
    """
    Takes a list of lists, all of which must be the same length,
    and returns their transpose.

    :param rows: List of lists, all must be the same length
    :returns: Transposed list of lists.
    """
    return list(map(list, zip(*rows)))


def list_to_csv(row):
    """
    Takes a list and converts it to a comma separated string.
    """

    format_string = ",".join(["{}"] * len(row))

    return format_string.format(*row)


def _convert_dict_to_maestro_params(samples):
    """Convert a scisample dictionary to a maestro dictionary"""
    keys = list(samples[0].keys())
    parameters = {}
    for key in keys:
        parameters[key] = {}
        parameters[key]["label"] = str(key) + ".%%"
        values = [sample[key] for sample in samples]
        parameters[key]["values"] = values
    return parameters


def find_duplicates(lst):
    """
    Takes a list and returns a list of any duplicate items.

    If there are no duplicates, return an empty list.
    Code taken from:
    https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    """
    seen = {}
    dupes = []

    for x in lst:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes
