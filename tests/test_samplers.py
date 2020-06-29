"""
Test module for testing code bucket structures.
"""

import os
import shutil
import tempfile
import unittest
import datetime
import pytest

from dirsetup.main import DirSetup, parse_paths

from scisample.utils import SamplingError
from scisample.samplers import (
    new_sampler,
    ListSampler,
    CrossProductSampler,
    CsvSampler,
    CustomSampler
    )
from scisample.utils import read_yaml

LIST_SAMPLER = """
sampler:
    type: list
    constants:
        X1: 20
    parameters:
        X2: [5, 10]
        X3: [5, 10]
"""

LIST_SAMPLER_WITH_DUPLICATES = """
sampler:
    type: list
    constants:
        X1: 20
    parameters:
        X1: [5, 10]
        X3: [5, 10]
        X3: [5, 10]
"""

CROSS_PRODUCT_SAMPLER = """
sampler:
    type: cross_product
    constants:
        X1: 20
    parameters:
        X2: [5, 10]
        X3: [5, 10]
"""

CSV_SAMPLER = """
sampler:
    type: csv
    csv_file: {path}/test.csv
    row_headers: True
"""

CSV1 = """X1,20,20
X2,5,10
X3,5,10"""

CUSTOM_SAMPLER = """
sampler:
    type: custom
    function: test_function
    module: {path}/codepy_sampler_test.py
    args:
        num_samples: 2
"""

CUSTOM_FUNCTION = """

def test_function(num_samples):
    return [{"X1": 20, "X2": 5, "X3": 5},
            {"X1": 20, "X2": 10, "X3": 10}][:num_samples]

"""

# @TODO: write BaseTestSampler
# @TODO: write test for LIST_SAMPLER_WITH_DUPLICATES
class TestListSampler(unittest.TestCase):
    """Unit test for testing the kernel bucket."""
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = LIST_SAMPLER
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")

        with open(self.sampler_file,'w') as _file:
            _file.write(self.definitions)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, ListSampler))

    def test_valid1(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['constants']
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['parameters']
        with pytest.raises(SamplingError) as excinfo:
            self.sampler.is_valid()
        assert ("Either constants or parameters must be included" 
            in str(excinfo.value))

    # scisample test with exception
    def test_valid2(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['parameters']
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['constants']
        with pytest.raises(SamplingError) as excinfo:
            self.sampler.is_valid()
        assert ("Either constants or parameters must be included" 
            in str(excinfo.value))

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)


class TestCrossProductSampler(unittest.TestCase):
    """Unit test for testing the kernel bucket."""
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = CROSS_PRODUCT_SAMPLER
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")

        with open(self.sampler_file,'w') as _file:
            _file.write(self.definitions)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, CrossProductSampler))

    def test_valid1(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['constants']
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['parameters']
        self.assertFalse(self.sampler.is_valid())

    def test_valid2(self):
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['parameters']
        self.assertTrue(self.sampler.is_valid())
        del self.sampler.data['constants']
        self.assertFalse(self.sampler.is_valid())

    def test_samples(self):
        samples = self.sampler.get_samples()
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


class TestCsvSampler(unittest.TestCase):
    """Unit test for testing the kernel bucket."""
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = CSV_SAMPLER.format(path=self.tmp_dir)
        self.csv_data = CSV1
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.csv_file = os.path.join(self.tmp_dir, "test.csv")
        with open(self.sampler_file,'w') as _file:
            _file.write(self.definitions)
        with open(self.csv_file,'w') as _file:
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
    """Unit test for testing the kernel bucket."""
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = CUSTOM_SAMPLER.format(path=self.tmp_dir)
        self.function_data = CUSTOM_FUNCTION
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.function_file = os.path.join(self.tmp_dir, "codepy_sampler_test.py")
        with open(self.sampler_file,'w') as _file:
            _file.write(self.definitions)
        with open(self.function_file,'w') as _file:
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
