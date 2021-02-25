import logging
import random
import os
import sys
from contextlib import suppress

# @TODO: can this duplicate code be removed?
UQPIPELINE_SAMPLE = False
UQPIPELINE_SAMPLE_PATH = '/collab/usr/gapps/uq/UQPipeline/smplg_cmpnt'
if os.path.exists(UQPIPELINE_SAMPLE_PATH):
    sys.path.append(UQPIPELINE_SAMPLE_PATH)
    with suppress(ModuleNotFoundError):
        import sampling.sampler as sampler
        import sampling.composite_samples as composite_samples
        UQPIPELINE_SAMPLE = True

from scisample.base_sampler import BaseSampler
from scisample.utils import log_and_raise_exception

from scisample.samplers import new_sampler 
import yaml 
def new_sampler_from_yaml(yaml_text):
    """Returns sampler from yaml text"""
    return new_sampler(
        yaml.safe_load(yaml_text))

# yaml_text = """
#     type: uqpipeline
#     uq_samples: my_samples
#     uq_code: |
#         my_samples = composite_samples.Samples()
#         my_samples.set_continuous_variable('density', 1.2, 2.8, 3.9)
#         my_samples.set_discrete_variable('material',
#                                  ['steel', 'aluminum'],
#                                  'steel')
#         my_samples.generate_samples(['density', 'material'],
#                             sampler.LatinHyperCubeSampler(),
#                             num_points=10)
#     """

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
print("yaml\n", yaml_text)
sci_sampler = new_sampler_from_yaml(yaml_text)


samples = sci_sampler.get_samples()
print("\nscisample.list:", samples)

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

yaml_text = """
    type: uqpipeline
    uq_points: points
    uq_variables: ['X1', 'X2']
    uq_code: |
        points = sampler.LatinHyperCubeSampler.sample_points(
            num_points=4, box=[[0, 1], [0, 1]])
    """
print(yaml_text)
sci_sampler = new_sampler_from_yaml(yaml_text)

samples = sci_sampler.get_samples()
print("\nscisample.list:", samples)


yaml_text = """
    type: list
    constants:
        X1: 20
    parameters:
        X2: [5, 10]
        X3: [5, 10]
    """

sci_sampler = new_sampler_from_yaml(yaml_text)

samples = sci_sampler.get_samples()
print("\nscisample.list:", samples)


my_samples = composite_samples.Samples()
my_samples.set_continuous_variable('density', 1.2, 2.8, 3.9)
my_samples.set_discrete_variable('material',
                                 ['steel', 'aluminum'],
                                 'steel')
my_samples.generate_samples(['density', 'material'],
                            sampler.LatinHyperCubeSampler(),
                            num_points=10)
print("my samples:", my_samples.get_points())

# UQPipelineSampler
name = "latin_hypercube"
my_sampler = sampler.LatinHyperCubeSampler
points = my_sampler.sample_points(num_points=10, box=[[0, 1], [0, 1]])
print(f"\n{my_sampler.name} points:\n{points}")
print(f"\n{my_sampler.name} dir:\n{dir(my_sampler)}")
points = my_sampler.sample_points(num_points=10, box=[[-1,1],[0,2]], geo_degree=1.3, seed=42)
print(f"\n{my_sampler.name} points:\n{points}")

name = "cartesian_cross"
my_sampler = sampler.CartesianCrossSampler
points = my_sampler.sample_points(box=[[0,1],[0,1]], num_divisions=3)
print(f"\n{my_sampler.name} points:\n{points}")
# returns 9 points, with 3 divisions in each dimension
points = my_sampler.sample_points(box=[[0,1],[0,1]], num_divisions=[3,4])
print(f"\n{my_sampler.name} points:\n{points}")
# returns 12 points, with 3 divisions in the first dimension, and 4 divisions in the second
points = my_sampler.sample_points(box=[[0,1],[0,1]], num_divisions=[[.5,.75],[.3,.32]])
print(f"\n{my_sampler.name} points:\n{points}")
points = my_sampler.sample_points(
    num_divisions=[3,3], 
    box=[[-1,1],[]], 
    values=[[],['foo', 'bar', 'zzyzx']])
print(f"\n{my_sampler.name} points:\n{points}")

name = "monte_carlo"
my_sampler = sampler.MonteCarloSampler
points = my_sampler.sample_points(num_points=6, box=[[-1,1],[0,2]], seed=42)
print(f"\n{my_sampler.name} points:\n{points}")


name= "uniform"
my_sampler = sampler.UniformSampler
points = my_sampler.sample_points(num_points=5, box=[[-1,1],[0,2]])
print(f"\n{my_sampler.name} points:\n{points}")

name = "quasi_random"
my_sampler = sampler.QuasiRandomNumberSampler()
points = my_sampler.sample_points(num_points=6, box=[[-1,1],[0,2]], technique='sobol')
print(f"\n{my_sampler.name} points (sobol):\n{points}")

my_sampler = sampler.QuasiRandomNumberSampler()
points = my_sampler.sample_points(num_points=6, box=[[-1,1],[0,2]], technique='halton')
print(f"\n{my_sampler.name} points (halton):\n{points}")

name = "corners"
my_sampler = sampler.CornerSampler()
points = my_sampler.sample_points(num_points=4, box=[[-1,1],[0,2]])
print(f"\n{my_sampler.name} points (halton):\n{points}")

name = "centered"
my_sampler = sampler.CenteredSampler()
points = my_sampler.sample_points(
    num_divisions=3, box=[[-1,1],[0,2]], dim_indices=[0,1], default=[0.5,0.5])
print(f"\n{my_sampler.name} points:\n{points}")

points = my_sampler.sample_points(
    num_divisions=3, box=[[-1,1],[0,2]], dim_indices=[0,1], technique='lhs_vals', num_points=3, seed=42)
print(f"\n{my_sampler.name} points (latin hyper cube):\n{points}")

name = "one_at_a_time"
my_sampler = sampler.OneAtATimeSampler()
points = my_sampler.sample_points(
    box=[[-1,1],[0,2]], default=[.5,.5], do_oat=True, use_high=True, use_low=True, use_default=True)
print(f"\n{my_sampler.name} points:\n{points}")
print("MOAT name not set correctly")
points = my_sampler.sample_points(
    box=[[-1,1],[0,2]], default=[.5,.5], do_oat=True, use_high=False, use_low=False, use_default=True)
print(f"\n{my_sampler.name} points (no high or low):\n{points}")
print("MOAT name not set correctly")

name = "faces"
my_sampler = sampler.FaceSampler()
points = my_sampler.sample_points(num_divisions=3, box=[[-1,1],[0,2]])
print(f"\n{my_sampler.name} points:\n{points}")
