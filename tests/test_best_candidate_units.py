"""
Sampler Tests

Unit tests for the BestCandidateSampler
"""

import os
import shutil
import tempfile
import unittest
import importlib
import pandas as pd

import yaml

from scisample.best_candidate_sampler import BestCandidateSampler
from scisample.samplers import new_sampler
from scisample.utils import SamplingError, read_yaml

PANDAS_PLUS = True
if (not importlib.util.find_spec("pandas")
        or not importlib.util.find_spec("numpy")
        or not importlib.util.find_spec("scipy.spatial")):
    PANDAS_PLUS = False

def new_sampler_from_yaml(yaml_text):
    """Returns sampler from yaml text"""
    return new_sampler(
        yaml.safe_load(yaml_text))

class TestBestCandidateUnits(unittest.TestCase):
    """
    Scenario: unit tests for BestCandidate
    """
    def setUp(self):
        if not PANDAS_PLUS:
            self.skipTest("Pandas not installed")
        yaml_text = """
            type: best_candidate
            num_samples: 5
            constants:
                X1: 20
                X2: foo
            parameters:
                X3:
                    min: 0
                    max: 25
                X4:
                    min: 0
                    max: 25
            """
        self.sampler = new_sampler_from_yaml(yaml_text)

    def test_manhattan_distance(self):
        """
        Given two points,
        manhattan_distance should return the correct distance.
        """

        assert self.sampler.distance((0, 0), (1, 1)) == 2
        assert self.sampler.distance((0, 0), (1, 2)) == 3
        assert self.sampler.distance((0, 0), (2, 1)) == 3
        assert self.sampler.distance((0, 0), (2, 2)) == 4
        assert self.sampler.distance((0, 0), (3, 3)) == 6

    def test_make_distance_map(self):
        """
        Given a set of points,
        make_distance_map should return the correct distance map.
        """
        # Sample list of points 1
        points = [
            {'x': 1, 'y': 1, 'z': 1},
            {'x': 2, 'y': 2, 'z': 2},
            {'x': 3, 'y': 3, 'z': 3},
        ]
        # Create DataFrame
        df = pd.DataFrame(points)
        distance_map, min_distance, max_distance = (
            self.sampler.make_distance_map(df, ['x', 'y']))
        print(distance_map, min_distance, max_distance)
        assert distance_map == {
            0: [[2, 1], [4, 2]],
            1: [[2, 0], [2, 2]],
            2: [[2, 1], [4, 0]]
        }
        assert min_distance == 2
        assert max_distance == 2
        # Sample list of points 2
        points = [
            {'x': 1, 'y': 1, 'z': 1},
            {'x': 2, 'y': 2, 'z': 2},
            {'x': 4, 'y': 4, 'z': 4},
        ]
        # Create DataFrame
        df = pd.DataFrame(points)
        distance_map, min_distance, max_distance = (
            self.sampler.make_distance_map(df, ['x', 'y']))
        print(distance_map, min_distance, max_distance)
        assert min_distance == 2
        assert max_distance == 4
        assert distance_map == {
            0: [[2, 1], [6, 2]],
            1: [[2, 0], [4, 2]],
            2: [[4, 1], [6, 0]]
        }
        # Sample list of points 3
        points = [
            {'x': 1, 'y': 1, 'z': 1},
            {'x': 2, 'y': 2, 'z': 2},
            {'x': 4, 'y': 4, 'z': 4},
        ]
        # Create DataFrame
        df = pd.DataFrame(points)
        distance_map, min_distance, max_distance = (
            self.sampler.make_distance_map(df, ['x', 'y', 'z']))
        print(distance_map, min_distance, max_distance)
        assert min_distance == 3
        assert max_distance == 6
        assert distance_map == {
            0: [[3, 1], [9, 2]],
            1: [[3, 0], [6, 2]],
            2: [[6, 1], [9, 0]]
        }

class TestDownselect(unittest.TestCase):
    """
    Scenario: unit tests for downselect
    """
    def setUp(self):
        if not PANDAS_PLUS:
            self.skipTest("Pandas not installed")
        yaml_text = """
            type: column_list
            parameters: |
                x     y
                1     1
                1.1   1.1
                2     2
                4     4
                8     8
            """
        self.sampler = new_sampler_from_yaml(yaml_text)

    def test_downselect_1(self):
        """
        Given 5 points, downselect(3) should remove nearest neighbors
        """
        # samples = self.sampler.get_samples()
        # print(samples)
        self.sampler.downselect(3)
        samples = self.sampler.get_samples()
        assert samples == [
            {'x': 1.0, 'y': 1.0},
            {'x': 8.0, 'y': 8.0},
            {'x': 4.0, 'y': 4.0},
            ]

    def test_downselect_2(self):
        """
        Given 5 points, downselect(3) should remove nearest neighbors
        """
        # samples = self.sampler.get_samples()
        # print(samples)
        previous_samples_list = [{'x': 1.0, 'y': 1.0}]
        previous_samples_df = pd.DataFrame(previous_samples_list)
        self.sampler.downselect(2, previous_samples=previous_samples_df)
        samples = self.sampler.get_samples()
        assert samples == [
            {'x': 8.0, 'y': 8.0},
            {'x': 4.0, 'y': 4.0},
            ]

    def test_downselect_3(self):
        """
        Given 5 points, downselect(3) should remove nearest neighbors
        """
        # samples = self.sampler.get_samples()
        # print(samples)
        previous_samples_list = [{'x': 1.2, 'y': 1.2}]
        previous_samples_df = pd.DataFrame(previous_samples_list)
        self.sampler.downselect(2, previous_samples=previous_samples_df)
        samples = self.sampler.get_samples()
        assert samples == [
            {'x': 8.0, 'y': 8.0},
            {'x': 4.0, 'y': 4.0},
            ]

class TestInterpolatePointsSymmetrically(unittest.TestCase):
    """
    Scenario: unit tests for interpolate_points_symmetrically
    """
    def setUp(self):
        if not PANDAS_PLUS:
            self.skipTest("Pandas not installed")
        self.sampler = None

    def test_interpolate_points_symmetrically_1(self):
        """
        Given 5 points, test_interpolate_points_symmetrically
        should add 4 points
        """
        yaml_text = """
            type: column_list
            parameters: |
                x     y
                1     1
                2     2
                4     4
                8     8
                16     16
            """
        self.sampler = new_sampler_from_yaml(yaml_text)
        previous_samples = pd.DataFrame(self.sampler.get_samples())
        yaml_text = """
            type: best_candidate
            num_samples: 5
            constants:
                X1: 20
                X2: foo
            parameters:
                x:
                    min: 0
                    max: 25
                y:
                    min: 0
                    max: 25
            """
        best_candidate_sampler = new_sampler_from_yaml(yaml_text)
        new_rows = best_candidate_sampler.interpolate_points_symmetrically(
            previous_samples, ['x', 'y'])
        assert len(new_rows) == 4
        new_values = [6.0, 10.0, 12.0, 14.0]
        for value in new_values:
            assert value in new_rows['x'].values
            assert value in new_rows['y'].values

    def test_interpolate_points_symmetrically_2(self):
        """
        Given 6 points, test_interpolate_points_symmetrically
        should add 1 point
        """
        yaml_text = """
            type: column_list
            parameters: |
                x     y
                1     1
                2     2
                4     4
                8     8
                16    16
                18    18
            """
        self.sampler = new_sampler_from_yaml(yaml_text)
        previous_samples = pd.DataFrame(self.sampler.get_samples())
        yaml_text = """
            type: best_candidate
            num_samples: 5
            constants:
                X1: 20
                X2: foo
            parameters:
                x:
                    min: 0
                    max: 25
                y:
                    min: 0
                    max: 25
            """
        best_candidate_sampler = new_sampler_from_yaml(yaml_text)
        new_rows = best_candidate_sampler.interpolate_points_symmetrically(
            previous_samples, ['x', 'y'])
        assert len(new_rows) == 1
        new_values = [6.0]
        for value in new_values:
            assert value in new_rows['x'].values
            assert value in new_rows['y'].values

class TestInterpolatePointsAsymmetrically(unittest.TestCase):
    """
    Scenario: unit tests for interpolate_points_asymmetrically
    """
    def setUp(self):
        if not PANDAS_PLUS:
            self.skipTest("Pandas not installed")
        self.sampler = None

    def test_interpolate_points_asymmetrically_1(self):
        """
        Given 4 points, test_interpolate_points_ssymmetrically
        should add 1 point
        """
        yaml_text = """
            type: column_list
            parameters: |
                x     y
                1     1
                2     2
                4     4
                8     8
                16    16
                18    18
            """
        self.sampler = new_sampler_from_yaml(yaml_text)
        previous_samples = pd.DataFrame(self.sampler.get_samples())
        yaml_text = """
            type: best_candidate
            num_samples: 5
            constants:
                X1: 20
                X2: foo
            parameters:
                x:
                    min: 0
                    max: 25
                y:
                    min: 0
                    max: 25
            """
        best_candidate_sampler = new_sampler_from_yaml(yaml_text)
        new_rows = best_candidate_sampler.interpolate_points_asymmetrically(
            previous_samples, ['x', 'y'])
        print(previous_samples)
        print(new_rows)
        assert len(new_rows) == 4
        # Points to check
        points = [
            [1, 0],
            [0, 1],
            [18, 20],
            [20, 18],
        ]
        for point in points:
            subset = new_rows[new_rows['x'] == point[0]]
            assert point[1] in subset['y'].values
