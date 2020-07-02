"""
Test module for testing scisampling classes and methods.
"""

import os
import shutil
import tempfile
import unittest
import pytest
import yaml

from scisample.utils import SamplingError
from scisample.samplers import (
    new_sampler,
    CsvSampler,
    CustomSampler
    )
from scisample.utils import read_yaml


def new_sampler_from_yaml(yaml_text):
    return new_sampler(
        yaml.safe_load(yaml_text))


class TestScisample(unittest.TestCase):
    """Unit test for testing several samplers."""

    def test_exceptions(self):
        """Unit test for testing invalid or unusual inputs."""

        yaml_text = """
            type: list
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with pytest.raises(SamplingError) as excinfo:
            sampler = new_sampler_from_yaml(yaml_text)
            sampler.is_valid()
        assert ("Either constants or parameters must be included"
                in str(excinfo.value))

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
        with pytest.raises(SamplingError) as excinfo:
            sampler = new_sampler_from_yaml(yaml_text)
            sampler.is_valid()
        assert (
            "The following constants or parameters are defined more than once"
            in str(excinfo.value))

        yaml_text = """
            type: list
            constants:
                X1: 20
            #parameters:
            #    X2: [5, 10]
            #    X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 1)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)

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

    def test_list_sampler(self):
        """Unit test for testing list sampler."""
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[0]['X4'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
        self.assertEqual(samples[1]['X4'], 10)

    def test_cross_product_sampler(self):
        """Unit test for testing cross product sampler."""
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

    def test_column_list_sampler(self):
        """Unit test for testing column list sampler."""
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

    def test_random_sampler(self):
        """Unit test for testing random sampler."""
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
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
            self.assertTrue(sample['X2'] > 5)
            self.assertTrue(sample['X3'] > 5)
            self.assertTrue(sample['X2'] < 10)
            self.assertTrue(sample['X3'] < 10)

    def test_best_candidate_sampler(self):
        """Unit test for testing best candidate sampler."""
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
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
            self.assertTrue(sample['X2'] > 5)
            self.assertTrue(sample['X3'] > 5)
            self.assertTrue(sample['X2'] < 10)
            self.assertTrue(sample['X3'] < 10)


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

    def test_valid1(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['csv_file']
        self.assertFalse(self.sampler.is_valid())

    def test_valid2(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['row_headers']
        self.assertFalse(self.sampler.is_valid())

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(sample['X1'], "20")
        self.assertEqual(samples[0]['X2'], "5")
        self.assertEqual(samples[0]['X3'], "5")
        self.assertEqual(samples[1]['X2'], "10")
        self.assertEqual(samples[1]['X3'], "10")


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

    def test_valid1(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['function']
        self.assertFalse(self.sampler.is_valid())

    def test_valid2(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['module']
        self.assertFalse(self.sampler.is_valid())

    def test_valid3(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['args']
        self.assertFalse(self.sampler.is_valid())

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
