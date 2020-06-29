"""
Module defining BaseSampler class.
"""

import logging

from jsonschema import ValidationError
from contextlib import suppress

from scisample.interface import SamplerInterface
from scisample.schema import validate_sampler
from scisample.utils import (
    list_to_csv, find_duplicates, _convert_dict_to_maestro_params
    )

# @TODO: can this duplicate code be removed?
MAESTROWF = False
with suppress(ModuleNotFoundError):
    from maestrowf.datastructures.core import ParameterGenerator
    MAESTROWF = True

LOG = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exceptions in this module."""
    # @TODO confirm that scisample exceptions are labelled clearly
    pass


class BaseSampler(SamplerInterface):
    """
    Base sampler class.
    """
    # @TODO: define SAMPLE_FUNCTIONS_DICT automatically:
    # https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-namedefine keywords
    # def all_subclasses(cls):
    #     return set(cls.__subclasses__()).union(
    #         [s for c in cls.__subclasses__() for s in all_subclasses(c)])

    SAMPLE_FUNCTIONS_DICT = {}
    SAMPLE_FUNCTIONS_KEYS = []
    """list: List of available sampling methods."""

    def __init__(self, data):
        """
        Initialize the sampler.

        :param data: Dictionary of sampler data.
        """
        self._data = data
        self._samples = None
        self._parameter_block = None
        self._pgen = None

    @property
    def data(self):
        """
        Sampler data block.
        """
        return self._data

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        :returns: True if the schema is valid, False otherwise.
        """
        LOG.info("BaseSampler.is_valid()")
        try:
            validate_sampler(self.data)
            return True
        except ValueError:
            LOG.error(f"No 'type' entry found in sampler data {self.data}")
        except KeyError:
            LOG.error(f"Sampler type {self.data['type']} not found in schema")
        except ValidationError:
            LOG.exception("Sampler data is invalid")

        return False

    def _check_parameters_constants_existence(self):
        if 'constants' not in self.data and 'parameters' not in self.data:
            msg = ("Either constants or parameters must be included in the " +
                   "sampler data")
            LOG.error(msg)
            raise Error(msg)

    def _check_parameters_constants_for_dups(self):
        LOG.info("testing for duplicates")
        if len(self.parameters) != len(set(self.parameters)):
            dupes = set(find_duplicates(self.parameters))
            msg = ("The following constants or parameters are defined more " +
                   "than once: " + str(dupes))
            LOG.error(msg)
            raise Error(msg)

    @property
    def parameter_block(self):
        """
        Converts samples to parameter dictionary for ``codepy setup`` and ``codepy run``

        The keys are the labels and the values are a string version of the
        list, so it can be easily passed to Jinja.
        """
        if self._parameter_block is None:
            self._parameter_block = {}
            for sample in self.get_samples():
                for key, value in sample.items():
                    if key not in self._parameter_block:
                        self._parameter_block[key] = []
                    self._parameter_block[key].append(value)

            for key, value in self._parameter_block.items():
                self._parameter_block[key] = list_to_csv(value)

        return self._parameter_block

    @property
    def maestro_pgen(self):
        """
        Returns a maestrowf Parameter Generator object containing samples
        """
        if not MAESTROWF:
            raise Exception("maestrowf is not installed\n" +
                            "the maestro_pgen is not supported")
        if self._pgen is not None:
            return self._pgen

        pgen = ParameterGenerator()
        params = _convert_dict_to_maestro_params(self.get_samples())

        for key, value in params.items():
            pgen.add_parameter(key, value["values"], value["label"])

        self._pgen = pgen
        return pgen
