"""
Module defining the UQ Pipeline sampler object.
"""

import logging
import random
import os
import sys
from contextlib import suppress

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
from scisample.utils import test_for_uniform_lengths, test_for_min_max

# @TODO: can this duplicate code be removed?

LOG = logging.getLogger(__name__)

class UQPipelineSampler(BaseSampler):
    """
    Class which wraps UQPipeline sampling methods.

    The UQPipeline is a proprietary LLNL uncertainty quantification tool. 
        * "Ensemble Calculation via the LLNL UQ Pipeline: A User's Guide", 
          Brandon, S; Christianson, G; Domyancic, D; Lucas, D; McEnerney, J; 
          Klein, R I; Tannahill, J; Lawrence Livermore National Laboratory, 2011
        * https://llnl.primo.exlibrisgroup.com/permalink/01LLNL_INST/1g1o79t/alma991000602089706316

    This class currently supports two ways of creating samples with 
    UQPipeline methods:

    1. With "sampler.<SAMPLER_NAME>.sample_points" methods, and
    2. with the "composite_samples" class.

    .. code:: yaml

         # `sample_points` method example.
         sampler:
            type: uqpipeline
            uq_points: points
            uq_variables: ['X1', 'type']
            uq_code: |
            points = sampler.CartesianCrossSampler.sample_points(
                num_divisions=[3,3], 
                box=[[-1,1],[]], 
                values=[[],['foo', 'bar']])

    results in the following sample set:

    .. code:: python

        [{'X1': -1.0, 'type': 'foo'}, 
         {'X1': -1.0, 'type': 'bar'}, 
         {'X1': 0.0, 'type': 'foo'}, 
         {'X1': 0.0, 'type': 'bar'}, 
         {'X1': 1.0, 'type': 'foo'}, 
         {'X1': 1.0, 'type': 'bar'}]

    .. code:: yaml

        # `composite_samples` method example.
        sampler:
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

    results in the same sample set:

    .. code:: python

        [{'X1': -1.0, 'type': 'foo'}, 
         {'X1': -1.0, 'type': 'bar'}, 
         {'X1': 0.0, 'type': 'foo'}, 
         {'X1': 0.0, 'type': 'bar'}, 
         {'X1': 1.0, 'type': 'foo'}, 
         {'X1': 1.0, 'type': 'bar'}]
    """
    def __init__(self, data):
        """
        Initialize the sampler.

        :param data: Dictionary of sampler data.
        """
        super().__init__(data)
        self.check_validity()

    def check_validity(self):
        code = 'uq_code' in self.data
        samples = 'uq_samples' in self.data
        points = 'uq_points' in self.data
        variables = 'uq_variables' in self.data
        if not code:
            log_and_raise_exception(
                "'uq_code' is required for a uqpipeline sampler.")
        if samples and not (points or variables):
            self._uq_type = "samples"
        if (points and variables) and not samples:
            self._uq_type = "points"
        if not (samples or points):
            log_and_raise_exception(
                "Either 'uq_samples' or 'uq_points' are required for a uqpipeline sampler.")
        if (samples and points):
            log_and_raise_exception(
                "Only 'uq_samples' or 'uq_points' can be specified for a uqpipeline sampler.")
        if (samples and variables):
            log_and_raise_exception(
                "Only 'uq_samples' or 'uq_variables' can be specified for a uqpipeline sampler.")
        if (points and not variables):
            log_and_raise_exception(
                "Both 'uq_points' and 'uq_variables' are required for a uqpipeline sampler.")

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        self.get_samples()
        return self._uq_parameters

    def get_samples(self):
        """
        Get samples from the sampler.

        This returns samples as a list of dictionaries, with the
        sample variables as the keys:

        .. code:: python

            [{'b': 0.89856, 'a': 1}, {'b': 0.923223, 'a': 1}, ... ]
        """
        # Note: I am being careful with internal variables
        #       to avoid conflicts with exec(uq_code)
        if self._samples is not None:
            return self._samples
        LOG.info("generating uqpipeline samples")
        try:
            exec(self.data['uq_code'])
        except:
            log_and_raise_exception(
                "unknown error when executing 'uq_code':"
                f"{self.data['uq_code']}")
        try:
            if self._uq_type == "samples":
                _uq_samples = eval(self.data['uq_samples'])
                _uq_variables = _uq_samples.get_variable_list()
                _uq_points = _uq_samples.get_points()
                self._uq_variables = _uq_variables
            elif self._uq_type == "points":
                _uq_points = eval(self.data['uq_points'])
                _uq_variables = self.data['uq_variables']
                self._uq_variables = _uq_variables
            else:
                log_and_raise_exception(
                    "Unknown error when attempting to use"
                    "uq_pipeline code with scisample."
                    "Please contact a developer.")
        except:
            log_and_raise_exception(
                "unknown error when evaluating 'uq_samples':"
                f"{self.data['uq_samples']}")

        self._samples = []

        _uq_parameters = {}
        for i in range(len(_uq_points[0])):
            _parameter_list = []
            for j in range(len(_uq_points)):
                _parameter_list.append(_uq_points[j][i])
            _uq_parameters[_uq_variables[i]] = _parameter_list
        
        with suppress(KeyError):
            for key, value in _uq_parameters.items():
                num_samples = len(value)
                break

        for i in range(num_samples):
            new_sample = {}

            with suppress(KeyError):
                new_sample.update(self.data['constants'])

            with suppress(KeyError):
                for key, value in _uq_parameters.items():
                    new_sample[key] = value[i]

            self._samples.append(new_sample)
        return self._samples

# # extra
#   points = sampler.LatinHyperCubeSampler.sample_points(
#                     num_points=10, box=[[0, 1], [0, 1]])

#             type: uqpipeline
#             uq_type: <UQPipeline Sampler keyword>
#                      cartesian_cross, centered, corners, default_value, geolhs,
#                      list, montecarlo, moat, multi_normal, pdf, quasi_rn,
#                      rawsamplepoints, samplepoints, stdlhs, uniform
#                      <Also accepts class names>
#                      LatinHyperCubeSampler, CartesianCrossSampler
#             num_samples: 5      # uq_type accepts either
#             num_points: 5       # uq_type accepts either
#             <uqpipeline parameters>
#             constants:
#                 X1: 20
#             parameters:         # uq_type box and range are entered here
#                 X2:             # some uq_types accept range or list
#                     min: 5
#                     max: 10
#                 X3: [5, 10]     # some uq_types accept range or list

#     A total of ``num_samples`` will be generated. Entries in the ``constants``
#     dictionary will be added to all samples. Entries in the ``parameters``
#     block will be selected from a range of ``min`` to ``max``.  The result of
#     the above block would something like: