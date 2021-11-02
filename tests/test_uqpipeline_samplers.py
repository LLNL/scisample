"""
UQPipeline Sampler Tests

Unit tests for  UQPipeline sampler objects:
"""

# @TODO: keep adding more samplers

import os
import sys
# import shutil
# import tempfile
import unittest
from contextlib import suppress

import pytest
import yaml

from scisample.base_sampler import BaseSampler
from scisample.utils import log_and_raise_exception
from scisample.uqpipeline_sampler import UQPipelineSampler

from scisample.samplers import new_sampler 

def new_sampler_from_yaml(yaml_text):
    """Returns sampler from yaml text"""
    return new_sampler(
        yaml.safe_load(yaml_text))

@pytest.mark.skipif(not UQPipelineSampler.UQPIPELINE_SAMPLE, reason="uqpipeline was not imported")
class TestScisampleUQPipeline(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for ListSampler
    """

    def test_class_cartesian_cross(self):
        """
        Given a cartesian_cross specification
        And I request a new sampler with the uqpipeline class interface
        Then I should get a uqpipeline sampler
        With appropriate values
        """
        yaml_text = """
            type: uqpipeline
            uq_samples: my_samples
            uq_code: |
                my_samples = composite_samples.Samples()
                my_samples.set_continuous_variable('X1', -1, 0, 1)
                my_samples.set_discrete_variable('type', ['foo', 'bar'], 'foo')
                my_samples.generate_samples(
                    ['X1', 'type'],
                    sampler.CartesianCrossSampler(),
                    num_divisions=[3,2])
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, UQPipelineSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 6)

        self.assertEqual(samples, 
            [{'X1': -1.0, 'type': 'foo'}, 
             {'X1': -1.0, 'type': 'bar'}, 
             {'X1': 0.0, 'type': 'foo'}, 
             {'X1': 0.0, 'type': 'bar'}, 
             {'X1': 1.0, 'type': 'foo'}, 
             {'X1': 1.0, 'type': 'bar'}])


    def test_method_cartesian_cross(self):
        """
        Given a cartesian_cross specification
        And I request a new sampler with the uqpipeline method interface
        Then I should get a uqpipeline sampler
        With appropriate values
        """
        yaml_text = """
            type: uqpipeline
            uq_points: points
            uq_variables: ['X1', 'type']
            uq_code: |
                points = sampler.CartesianCrossSampler.sample_points(
                num_divisions=[3,2],
                box=[[-1,1],[]], 
                values=[[],['foo', 'bar']])
            """

        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, UQPipelineSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 6)

        self.assertEqual(samples, 
            [{'X1': -1.0, 'type': 'foo'}, 
             {'X1': -1.0, 'type': 'bar'}, 
             {'X1': 0.0, 'type': 'foo'}, 
             {'X1': 0.0, 'type': 'bar'}, 
             {'X1': 1.0, 'type': 'foo'}, 
             {'X1': 1.0, 'type': 'bar'}])

    def make_samples_test_results(self, method_call, expected_results):
        """
        Given a uqsampler method call and expected results
        And I request a new sampler
        Then I should get a uqpipeline sampler
        With appropriate values
        """
        yaml_text = f"""
            type: uqpipeline
            uq_points: points
            uq_variables: ['X1', 'X2']
            uq_code: |
                points = {method_call}
            """  
        sampler = new_sampler_from_yaml(yaml_text)
        print(f"method_call: {method_call}")    
        print(f"expected_results: {expected_results}")    
        samples = sampler.get_samples()
        print(f"samples: {samples}")    

        self.assertTrue(isinstance(sampler, UQPipelineSampler))
        self.assertEqual(len(samples), len(expected_results))
        self.assertEqual(samples, expected_results)

    def test_methods_uq_pipeline(self):
        """
        Given multiple uqpipeline specifications
        And I request a new sampler with the uqpipeline method interface
        Then I should get uqpipeline samplers
        With appropriate values
        """

        methods_and_results = [
            {'method_call': ("sampler.LatinHyperCubeSampler.sample_points("
                             "    num_points=4, box=[[0, 1], [0, 1]], seed=7)"),
             'expected_results': [{'X1': 0.5797430564433508, 'X2': 0.018012783339940386},
                 {'X1': 0.4945557242696456, 'X2': 0.3171097450254678},
                 {'X1': 0.11389622833205296, 'X2': 0.62497062520639},
                 {'X1': 0.8270031912969349, 'X2': 0.9198074990302352}]},
            {'method_call': ("sampler.MonteCarloSampler.sample_points("
                             "    num_points=6, box=[[-1,1],[0,2]], seed=42)"),
             'expected_results': [{'X1': -0.250919762305275, 'X2': 0.11616722433639892},
                 {'X1': 0.9014286128198323, 'X2': 1.7323522915498704},
                 {'X1': 0.4639878836228102, 'X2': 1.2022300234864176},
                 {'X1': 0.1973169683940732, 'X2': 1.416145155592091},
                 {'X1': -0.687962719115127, 'X2': 0.041168988591604894},
                 {'X1': -0.6880109593275947, 'X2': 1.9398197043239886}]},
            {'method_call': ("sampler.UniformSampler.sample_points("
                             "    num_points=5, box=[[-1,1],[0,2]])"),
             'expected_results': [{'X1': -1.0, 'X2': 0.0}, 
                 {'X1': -0.5, 'X2': 0.5}, 
                 {'X1': 0.0, 'X2': 1.0}, 
                 {'X1': 0.5, 'X2': 1.5}, 
                 {'X1': 1.0, 'X2': 2.0}]},
            {'method_call': ("sampler.QuasiRandomNumberSampler.sample_points("
                             "    num_points=4, box=[[-1,1],[0,2]], technique='sobol')"),
             'expected_results': [{'X1': 0.0, 'X2': 1.0}, 
                 {'X1': 0.5, 'X2': 0.5}, 
                 {'X1': -0.5, 'X2': 1.5}, 
                 {'X1': -0.25, 'X2': 0.75}]},
            {'method_call': ("sampler.CenteredSampler.sample_points("
                             "    num_divisions=3, box=[[-1,1],[0,2]]," 
                             "    dim_indices=[0,1], default=[0.5,0.5])"),
             'expected_results': [{'X1': -1.0, 'X2': 0.5}, 
                {'X1': 0.0, 'X2': 0.5}, 
                {'X1': 1.0, 'X2': 0.5}, 
                {'X1': 0.5, 'X2': 0.0}, 
                {'X1': 0.5, 'X2': 1.0}, 
                {'X1': 0.5, 'X2': 2.0}]},
            {'method_call': ("sampler.OneAtATimeSampler.sample_points("
                             "    box=[[-1,1],[0,2]], default=[.5,.5], "
                             "    do_oat=True, use_high=True, use_low=True," 
                             "    use_default=True)"),
             'expected_results': [{'X1': -1.0, 'X2': 0.0}, 
                {'X1': 1.0, 'X2': 2.0}, 
                {'X1': 0.5, 'X2': 0.5}, 
                {'X1': -1.0, 'X2': 0.5}, 
                {'X1': 1.0, 'X2': 0.5}, 
                {'X1': 0.5, 'X2': 0.0}, 
                {'X1': 0.5, 'X2': 2.0}]}
            ]
        for method_and_result in methods_and_results:
            self.make_samples_test_results(
                method_and_result['method_call'], 
                method_and_result['expected_results'])


