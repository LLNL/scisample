"""
Helper functions for ``scisample``.
"""

import csv
import logging
from contextlib import suppress

from cached_property import cached_property
import yaml
import numpy
import parse

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


def test_for_min_max(parameters):
    """Test for required `min` and `max` dictionary entries."""
    for key, value in parameters.items():
        try:
            float(value['min'])
        except ValueError:
            log_and_raise_exception(
                f"Parameter ({key}) must have a numeric minimum.\n"
                f"  Current minimum value is: {value}.")
        try:
            float(value['max'])
        except ValueError:
            log_and_raise_exception(
                f"Parameter ({key}) must have a numeric maximum.\n"
                f"  Current maximum value is: {value}.")


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


class ParameterMixIn:
    """
    Mixin for reading the different ways to define parameters.
    """
    @cached_property
    def _parsed_parameters(self):
        """
        Property containing the parsed parameters.
        """
        return {
            key: parse_parameters(value)
            for key, value in self.data['parameters'].items()
        }


def parse_parameters(data):
    """
    Takes a specification for a list of parameters and converts it to the list.

    If a list is passed, the list will be returned.

    If a dict is passed with the keys  "start" or "min", "stop" or "max",
        "step" or "num_points", a list will be constructed based on these
        parameters.

    If a string is passed, either of the form ``[start:stop:step]`` or
        ``start to stop by step``, a list will be constructed.

    :param data: Data to parse.
    :returns: List of parameters.
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        start = data.get('start', data.get('min'))
        stop = data.get('stop', data.get('max'))
        step = data.get('step')
        num_points = data.get('num_points')
        if start is None or stop is None:
            raise SamplingError(
                "Parameter dictionaries must define start and stop")
        return parameter_list(start, stop, step, num_points)

    if isinstance(data, str):
        parse_formats = [
            '{start} to {stop} by {step}',
            '[{start}:{stop}:{step}]'
        ]
        for fmt in parse_formats:
            result = parse.parse(fmt, data)
            if result:
                kwargs = {
                    key: float(value)
                    for key, value in result.named.items()
                }
                return parameter_list(**kwargs)
    raise SamplingError(f"Unable to parse parameters from {data}")


def parameter_list(start, stop, step=None, num_points=None):
    """
    Create a list of points.

    :param start: First point in the list
    :param stop: Last point in the list
    :param step: Step size.  Will be ignored if ``num_points`` is
        specified.
    :param num_points: Number of points.
    """
    if not step and not num_points:
        raise SamplingError("Must specify either number of points or step")

    if step and not num_points:
        return_list = list(numpy.arange(start, stop, step))
        return_list.append(stop)
    else:
        return_list = list(numpy.linspace(start, stop, num_points))

    return return_list
