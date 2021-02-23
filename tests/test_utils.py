"""
Tests for utility functions.
"""

from scisample.utils import parse_parameters, parameter_list


mylist = [1.0, 2.0, 3.0, 4.0, 5.0]

def test_parameter_list_step():
    """
    When I request a parameter list using steps,
    It should match the expected result.
    """
    assert parameter_list(start=1.0, stop=5.0, step=1.0) == mylist


def test_parameter_list_points():
    """
    When I request a parameter list using number of points,
    It should match the expected result.
    """
    assert parameter_list(start=1.0, stop=5.0, num_points=5) == mylist


def test_list():
    """
    Given a list of points,
    parse_parameters should return the same list.
    """
    assert parse_parameters(mylist) == mylist


def test_dict_step():
    """
    Given a dict containing start, stop, and step,
    parse_parameters should return the correct list.
    """
    assert parse_parameters({'start': 1.0, 'stop': 5.0, 'step': 1.0}) == mylist


def test_dict_num_points():
    """
    Given a dict containing min, max, and num_points,
    parse_parameters should return the correct list.
    """
    assert parse_parameters({'min': 1.0, 'max': 5.0, 'num_points': 5}) == mylist


def test_str_range():
    """
    Given a string ``[start:stop:step]``,
    parse_parameters should return the correct list.
    """
    assert parse_parameters("[1.0:5.0:1.0]") == mylist


def test_str_by():
    """
    Given a string ``start to stop by step``,
    parse_parameters should return the correct list.
    """
    assert parse_parameters("1.0 to 5.0 by 1.0") == mylist
