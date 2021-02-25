"""
Sampler Tests

Unit tests for the codepy sampler objects:
#. ListSampler,
#. ColumnListSampler,
#. CrossProductSampler,
#. RandomSampler,
#. BestCandidateSampler,
#. CsvSampler,
#. CustomSampler

More tests on invalid samplers are likely needed.
"""

import os
import shutil
import tempfile
import unittest
from contextlib import suppress

import pytest
import yaml

from scisample.best_candidate_sampler import BestCandidateSampler
from scisample.column_list_sampler import ColumnListSampler
from scisample.cross_product_sampler import CrossProductSampler
from scisample.list_sampler import ListSampler
from scisample.random_sampler import RandomSampler
from scisample.custom_sampler import CustomSampler
from scisample.csv_sampler import CsvSampler
from scisample.samplers import new_sampler
from scisample.utils import SamplingError, read_yaml #, new_sampler_from_yaml

PANDAS_PLUS = False
with suppress(ModuleNotFoundError):
    import pandas as pd
    import numpy as np
    import scipy.spatial as spatial
    PANDAS_PLUS = True

# @TODO: improve coverage

def new_sampler_from_yaml(yaml_text):
    """Returns sampler from yaml text"""
    return new_sampler(
        yaml.safe_load(yaml_text))

class TestScisampleExceptions(unittest.TestCase):
    """
    Scenario: Requesting samplers with invalid yaml input
    """

    def test_missing_type_exception(self):
        """
        Given a missing sampler type,
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            foo: bar
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "No type entry in sampler data"
            in str(context.exception))

    def test_invalid_type_exception(self):
        """
        Given an invalid sampler type,
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: foobar
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "not a recognized sampler type"
            in str(context.exception))

    def test_missing_data_exception(self):
        """
        Given no constants or parameters
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: list
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "Either constants or parameters must be included"
            in str(context.exception))

    def test_duplicate_data_exception(self):
        """
        Given a variable in both constants and parameters
        And I request a new sampler
        Then I should get a SamplerException
        """
        # @TODO: We can not detect if parameters are defined twice.
        # @TODO: Fixing this requires a rewrite of read_yaml.
        yaml_text = """
            type: list
            constants:
                X2: 20
            parameters:
                X2: [5, 10]
                X2: [5, 10]
                X3: [5, 10]
                X3: [5, 10]
             """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "The following constants or parameters are defined more than once"
            in str(context.exception))


class TestScisampleUniversal(unittest.TestCase):
    """
    Scenario: Testing behavior valid for multiple samplers
    """
    def test_constants_only(self):
        """
        Given only constants
        And I request a new sampler
        Then I should get a sampler with one sample
        With appropriate values
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
                X2: 30
            #parameters:
            #    X2: [5, 10]
            #    X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 1)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
            self.assertEqual(sample['X2'], 30)

    def test_parameters_only(self):
        """
        Given only parameters
        And I request a new sampler
        Then I should get appropriate values
        """
        yaml_text = """
            type: list
            #constants:
            #    X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)

        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)


class TestScisampleList(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for ListSampler
    """

    def test_normal(self):
        """
        Given a list specification
        And I request a new sampler
        Then I should get a ListSampler
        With appropriate values
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3:
                    min: 5
                    max: 10
                    step: 5
                X4: 5.0 to 10 by 5.0
                X5: "[5.0:10.0:5]"
                X6:
                    start: 5
                    stop: 10
                    num_points: 2
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, ListSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        for i in range(2, 7):
            self.assertEqual(samples[0][f'X{i}'], 5)
            self.assertEqual(samples[1][f'X{i}'], 10)
        sampler
        self.assertEqual(samples, 
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5, 'X5': 5, 'X6': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10, 'X5': 10, 'X6': 10}])
        self.assertEqual(sampler.parameter_block, 
            {'X1': {'values': [20, 20], 'label': 'X1.%%'}, 
             'X2': {'values': [5, 10], 'label': 'X2.%%'}, 
             'X3': {'values': [5, 10], 'label': 'X3.%%'}, 
             'X4': {'values': [5, 10], 'label': 'X4.%%'}, 
             'X5': {'values': [5, 10], 'label': 'X5.%%'}, 
             'X6': {'values': [5, 10], 'label': 'X6.%%'}})

    def test_error(self):
        """
        Given an invalid list specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10, 20]
                X3: [5, 10]
                X4: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "All parameters must have the same number of values"
            in str(context.exception))


class TestScisampleCrossProduct(unittest.TestCase):
    """
    Scenario: normal tests for CrossProductSampler
    """
    def test_normal(self):
        """
        Given a cross_product specification
        And I request a new sampler
        Then I should get a CrossProductSampler
        With appropriate values
        """
        yaml_text = """
            # sampler:
                type: cross_product
                constants:
                    X1: 20
                parameters:
                    X2: [5, 10]
                    X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, CrossProductSampler))

        samples = sampler.get_samples()

        self.assertEqual(sampler.parameters, ["X1", "X2", "X3"])
        self.assertEqual(len(samples), 4)

        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 5)
        self.assertEqual(samples[1]['X3'], 10)
        self.assertEqual(samples[2]['X2'], 10)
        self.assertEqual(samples[2]['X3'], 5)
        self.assertEqual(samples[3]['X2'], 10)
        self.assertEqual(samples[3]['X3'], 10)


class TestScisampleColumnList(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for ColumnListSampler
    """
    def test_normal(self):
        """
        Given a column_list specification
        And I request a new sampler
        Then I should get a ColumnListSampler
        With appropriate values
        """
        yaml_text = """
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2     X3     X4
                5      5      5
                10     10     10
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, ColumnListSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], '5')
        self.assertEqual(samples[0]['X3'], '5')
        self.assertEqual(samples[0]['X4'], '5')
        self.assertEqual(samples[1]['X2'], '10')
        self.assertEqual(samples[1]['X3'], '10')
        self.assertEqual(samples[1]['X4'], '10')

    def test_comments(self):
        """
        Given a column_list specification
        And I request a new sampler
        Then I should get a ColumnListSampler
        With appropriate values
        And any commented lines should be ignored.
        """
        yaml_text = """
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2     X3     X4
                5      5      5 # This is a comment
                10     10     10
                #15    15     15 # Don't process this line
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, ColumnListSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], '5')
        self.assertEqual(samples[0]['X3'], '5')
        self.assertEqual(samples[0]['X4'], '5')
        self.assertEqual(samples[1]['X2'], '10')
        self.assertEqual(samples[1]['X3'], '10')
        self.assertEqual(samples[1]['X4'], '10')

    def test_error(self):
        """
        Given an invalid column_list specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2     X3     X4
                5      5      5
                10     10     10
                20
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "All rows must have the same number of values"
            in str(context.exception))


class TestScisampleRandomSampler(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for RandomSampler
    """
    def test_normal(self):
        """
        Given a random specification
        And I request a new sampler
        Then I should get a RandomSampler
        With appropriate values
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5
                    max: 10
                X3:
                    min: 5
                    max: 10
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, RandomSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
            self.assertTrue(sample['X2'] > 5)
            self.assertTrue(sample['X3'] > 5)
            self.assertTrue(sample['X2'] < 10)
            self.assertTrue(sample['X3'] < 10)

    def test_normal2(self):
        """
        Given a random specification
        And I request a new sampler
        Then I should get a RandomSampler
        With appropriate values
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 0.5
            parameters:
                X2:
                    min: 0.2
                    max: 0.8
                X3:
                    min: 0.2
                    max: 0.8
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, RandomSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample['X1'], 0.5)
            self.assertTrue(sample['X2'] > 0.2)
            self.assertTrue(sample['X3'] > 0.2)
            self.assertTrue(sample['X2'] < 0.8)
            self.assertTrue(sample['X3'] < 0.8)

    def test_error1(self):
        """
        Given an invalid random specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
            parameters:
                X2:
                    min: foo
                    max: 10
                X3:
                    min: 5
                    max: 10
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "must have a numeric minimum"
            in str(context.exception))

    def test_error2(self):
        """
        Given an invalid random specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
            parameters:
                X2:
                    min: 1
                    max: bar
                X3:
                    min: 5
                    max: 10
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "must have a numeric maximum"
            in str(context.exception))

    def test_error3(self):
        """
        Given previous_samples
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: random
            num_samples: 5
            previous_samples: samples.csv 
            constants:
                X1: 20
            parameters:
                X2:
                    min: 1
                    max: bar
                X3:
                    min: 5
                    max: 10
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "'previous_samples' is not yet supported"
            in str(context.exception))


class TestScisampleBestCandidate(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for BestCandidate
    """
    def test_normal(self):
        """
        Given a best_candidate specification
        And I request a new sampler
        Then I should get a BestCandidate
        With appropriate values
        """
        yaml_text = """
            type: best_candidate
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5
                    max: 10
                X3:
                    min: 5
                    max: 10
            """
        if PANDAS_PLUS:
            sampler = new_sampler_from_yaml(yaml_text)
            self.assertTrue(isinstance(sampler, BestCandidateSampler))
            samples = sampler.get_samples()

            self.assertEqual(len(samples), 5)
            for sample in samples:
                self.assertEqual(sample['X1'], 20)
                self.assertTrue(sample['X2'] > 5)
                self.assertTrue(sample['X3'] > 5)
                self.assertTrue(sample['X2'] < 10)
                self.assertTrue(sample['X3'] < 10)
        else:
            # test only works if pandas is installed
            self.assertTrue(True)


class TestCsvSampler(unittest.TestCase):
    """Unit test for testing the csv sampler."""
    CSV_SAMPLER = """
    sampler:
        type: csv
        csv_file: {path}/test.csv
        row_headers: True
    """

    # Note: the csv reader does not ignore blank lines
    CSV1 = """X1,20,20
    X2,5,10
    X3,5,10"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = self.CSV_SAMPLER.format(path=self.tmp_dir)
        self.csv_data = self.CSV1
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.csv_file = os.path.join(self.tmp_dir, "test.csv")
        with open(self.sampler_file, 'w') as _file:
            _file.write(self.definitions)
        with open(self.csv_file, 'w') as _file:
            _file.write(self.csv_data)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))
        self.assertTrue(os.path.isfile(self.csv_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, CsvSampler))

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)

class TestCustomSampler(unittest.TestCase):
    """Unit test for testing the custom sampler."""

    CUSTOM_SAMPLER = """
        sampler:
            type: custom
            function: test_function
            module: {path}/codepy_sampler_test.py
            args:
                num_samples: 2
    """

    CUSTOM_FUNCTION = (
        """def test_function(num_samples):
               return [{"X1": 20, "X2": 5, "X3": 5},
                       {"X1": 20, "X2": 10, "X3": 10}][:num_samples]
        """)

    def setUp(self):
        print("CUSTOM_FUNCTION:\n" + self.CUSTOM_FUNCTION)
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = self.CUSTOM_SAMPLER.format(path=self.tmp_dir)
        self.function_data = self.CUSTOM_FUNCTION
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.function_file = os.path.join(self.tmp_dir,
                                          "codepy_sampler_test.py")
        with open(self.sampler_file, 'w') as _file:
            _file.write(self.definitions)
        with open(self.function_file, 'w') as _file:
            _file.write(self.function_data)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))
        self.assertTrue(os.path.isfile(self.function_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, CustomSampler))

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
