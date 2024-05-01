"""
Tests for utility functions.
"""

from scisample.utils import (
    parse_parameters, parameter_list, manhattan_distance)
import numpy as np
import pandas as pd

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
    assert (
        parse_parameters({'min': 1.0, 'max': 5.0, 'num_points': 5}) == mylist)


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


def test_manhattan_distance():
    """
    Given two points,
    manhattan_distance should return the correct distance.
    """
    assert manhattan_distance((0, 0), (1, 1)) == 2
    assert manhattan_distance((0, 0), (1, 2)) == 3
    assert manhattan_distance((0, 0), (2, 1)) == 3
    assert manhattan_distance((0, 0), (2, 2)) == 4
    assert manhattan_distance((0, 0), (3, 3)) == 6

    point1 = {
        'X': 1.2276834976629671,
        'Y': 1.5256709542398117,
        'Unnamed: 0.3': 127.0,
        'Z': 0.0859323834696076,
        'Unnamed: 0': np.nan,
        'Unnamed: 0.1': np.nan,
        'Unnamed: 0.2': np.nan
        }
    point2 = {
        'X': 2.2276834976629671,
        'Y': 1.5256709542398117,
        'Unnamed: 0.3': 127.0,
        'Z': 0.0859323834696076,
        'Unnamed: 0': np.nan,
        'Unnamed: 0.1': np.nan,
        'Unnamed: 0.2': np.nan
        }
    df = pd.DataFrame([point1, point2])
    assert manhattan_distance(df.iloc[0], df.iloc[1]) == 1.0

