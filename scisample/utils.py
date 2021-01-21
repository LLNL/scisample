"""
Helper functions for ``scisample``.
"""

import csv
import logging
from contextlib import suppress

import yaml

LOG = logging.getLogger(__name__)


class SamplingError(Exception):
    """Base class for exceptions in this module."""


def log_and_raise_exception(msg):
    """ Log error and raise exception """
    LOG.error(msg)
    raise SamplingError(msg)


def test_for_uniform_lengths(iterable):
    """ Test that each item in iterable is the same length """
    test_length = None
    for key, value in iterable:
        if test_length is None:
            test_key = key
            test_value = value
            test_length = len(value)
        if len(value) != test_length:
            log_and_raise_exception(
                "All parameters must have the " +
                "same number of values.\n"
                f"  Parameter ({test_key}) has {test_length} value(s):\n"
                f"    {test_value}.\n"
                f"  Parameter ({key}) has {len(value)} value(s):\n"
                f"    {value}.\n")


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
    results = []
    with open(filename, newline='') as _file:
        csvreader = csv.reader(
            _file,
            skipinitialspace=True,
            )
        for row in csvreader:
            new_row = []
            for tok in row:
                if tok.startswith('#'):
                    continue
                tok = tok.strip()
                with suppress(ValueError):
                    tok = float(tok)
                new_row.append(tok)
            if new_row:
                results.append(new_row)
    return results


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


def find_duplicates(items):
    """
    Takes a list and returns a list of any duplicate items.

    If there are no duplicates, return an empty list.
    Code taken from:
    https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    """
    seen = {}
    duplicates = []

    for item in items:
        if item not in seen:
            seen[item] = 1
        else:
            if seen[item] == 1:
                duplicates.append(item)
            seen[item] += 1
    return duplicates
