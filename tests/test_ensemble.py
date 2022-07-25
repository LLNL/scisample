"""
Unit tests for the ``Ensemble`` object.
"""

import os
import shutil
import tempfile
import unittest
from contextlib import suppress

import pytest
import yaml

from scisample.ensemble import Ensemble
from scisample.utils import SamplingError


class TestEnsemble(unittest.TestCase):
    """
    Scenario: using an ensemble object to combine multiple samplers.
    """
    def test_empty_ensemble(self):
        """
        If I request an Ensemble object,
        but include no sampler data,
        Then a ``SamplingError`` should be raised. 
        """
        with self.assertRaises(SamplingError) as context:
            ensemble = Ensemble()
        
        self.assertIn(
            "No samplers requested for ensemble",
            str(context.exception)
        )

    def test_invalid_subsampler(self):
        """
        Given a variable in both constants and parameters
        And I request a new sampler via an Ensemble,
        Then I should get a SamplerException
        """
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
            Ensemble(yaml.safe_load(yaml_text))
        self.assertIn(
            "The following constants or parameters are defined more than once",
            str(context.exception)
            )

    def test_mismatched_parameters(self):
        """
        Given two samplers which do not have the same parameters,
        And I request them via an Ensemble,
        Then I should get a Sampler Exception
        """
        yaml_text1 = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
             """
        yaml_text2 = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
             """
        with self.assertRaises(SamplingError) as context:
            Ensemble(yaml.safe_load(yaml_text1), yaml.safe_load(yaml_text2))
        self.assertIn(
            "All samplers in an ensemble must have the same",
            str(context.exception)
            )

    def test_add_sampler_mismatched_parameters(self):
        """
        Given two samplers which do not have the same parameters,
        And I add them both to an Ensemble,
        Then I should get a Sampler Exception
        """
        yaml_text1 = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
             """
        yaml_text2 = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
             """
        ensemble = Ensemble(yaml.safe_load(yaml_text1))
        with self.assertRaises(SamplingError) as context:
            ensemble.add_samplers(yaml.safe_load(yaml_text2))
        self.assertIn(
            "All samplers in an ensemble must have the same",
            str(context.exception)
            )

    def test_single_sampler(self):
        """
        Given an Ensemble with only one sampler,
        It should behave identically to the sampler.
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
            """
        sampler = Ensemble(yaml.safe_load(yaml_text))

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

        self.assertEqual(samples, 
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10}])
        self.assertEqual(sampler.parameter_block, 
            {'X1': {'values': [20, 20], 'label': 'X1.%%'}, 
             'X2': {'values': [5, 10], 'label': 'X2.%%'}, 
             'X3': {'values': [5, 10], 'label': 'X3.%%'}, 
             'X4': {'values': [5, 10], 'label': 'X4.%%'}})

    def test_multi_sampler(self):
        """
        Given an Ensemble with multiple samplers,
        It should return the combined samples from both samplers..
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
            """
        sampler = Ensemble(yaml.safe_load(yaml_text), yaml.safe_load(yaml_text))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 4)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[0]['X4'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
        self.assertEqual(samples[1]['X4'], 10)
        self.assertEqual(samples[2]['X2'], 5)
        self.assertEqual(samples[2]['X3'], 5)
        self.assertEqual(samples[2]['X4'], 5)
        self.assertEqual(samples[3]['X2'], 10)
        self.assertEqual(samples[3]['X3'], 10)
        self.assertEqual(samples[3]['X4'], 10)

        self.assertEqual(samples, 
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10},
             {'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10}])
        self.assertEqual(sampler.parameter_block, 
            {'X1': {'values': [20, 20, 20, 20], 'label': 'X1.%%'}, 
             'X2': {'values': [5, 10, 5, 10], 'label': 'X2.%%'}, 
             'X3': {'values': [5, 10, 5, 10], 'label': 'X3.%%'}, 
             'X4': {'values': [5, 10, 5, 10], 'label': 'X4.%%'}})

    def test_multi_sampler_list(self):
        """
        Given an Ensemble with multiple samplers,
        Initialized from a list  of samplers,
        It should return the combined samples from both samplers..
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
            """
        sampler = Ensemble([yaml.safe_load(yaml_text), yaml.safe_load(yaml_text)])

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 4)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[0]['X4'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
        self.assertEqual(samples[1]['X4'], 10)
        self.assertEqual(samples[2]['X2'], 5)
        self.assertEqual(samples[2]['X3'], 5)
        self.assertEqual(samples[2]['X4'], 5)
        self.assertEqual(samples[3]['X2'], 10)
        self.assertEqual(samples[3]['X3'], 10)
        self.assertEqual(samples[3]['X4'], 10)

        self.assertEqual(samples, 
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10},
             {'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10}])
        self.assertEqual(sampler.parameter_block, 
            {'X1': {'values': [20, 20, 20, 20], 'label': 'X1.%%'}, 
             'X2': {'values': [5, 10, 5, 10], 'label': 'X2.%%'}, 
             'X3': {'values': [5, 10, 5, 10], 'label': 'X3.%%'}, 
             'X4': {'values': [5, 10, 5, 10], 'label': 'X4.%%'}})

    def test_multi_sampler_add(self):
        """
        Given an Ensemble with one sampler,
        If I add another sampler,
        It should return the combined samples from both samplers..
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
                X4: [5, 10]
            """
        sampler = Ensemble(yaml.safe_load(yaml_text))

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

        self.assertEqual(samples, 
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10}])
        self.assertEqual(sampler.parameter_block, 
            {'X1': {'values': [20, 20], 'label': 'X1.%%'}, 
             'X2': {'values': [5, 10], 'label': 'X2.%%'}, 
             'X3': {'values': [5, 10], 'label': 'X3.%%'}, 
             'X4': {'values': [5, 10], 'label': 'X4.%%'}})

        sampler.add_samplers(yaml.safe_load(yaml_text))
        self.assertEqual(
            sampler._samples,
            None
        )

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 4)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[0]['X4'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
        self.assertEqual(samples[1]['X4'], 10)
        self.assertEqual(samples[2]['X2'], 5)
        self.assertEqual(samples[2]['X3'], 5)
        self.assertEqual(samples[2]['X4'], 5)
        self.assertEqual(samples[3]['X2'], 10)
        self.assertEqual(samples[3]['X3'], 10)
        self.assertEqual(samples[3]['X4'], 10)

        self.assertEqual(samples, 
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10},
             {'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10}])
        self.assertEqual(sampler.parameter_block, 
            {'X1': {'values': [20, 20, 20, 20], 'label': 'X1.%%'}, 
             'X2': {'values': [5, 10, 5, 10], 'label': 'X2.%%'}, 
             'X3': {'values': [5, 10, 5, 10], 'label': 'X3.%%'}, 
             'X4': {'values': [5, 10, 5, 10], 'label': 'X4.%%'}})
