"""
Module defining the BaseSampler class.
"""

import logging

from jsonschema import ValidationError
from contextlib import suppress

from scisample.interface import SamplerInterface
from scisample.schema import validate_sampler
from scisample.utils import (
    list_to_csv, find_duplicates, _convert_dict_to_maestro_params,
    log_and_raise_exception
    )

# @TODO: can this duplicate code be removed?
MAESTROWF = False
with suppress(ModuleNotFoundError):
    from maestrowf.datastructures.core import ParameterGenerator
    MAESTROWF = True

PANDAS_PLUS = False
with suppress(ModuleNotFoundError):
    import pandas as pd
    import numpy as np
    import scipy.spatial as spatial
    PANDAS_PLUS = True

LOG = logging.getLogger(__name__)


class BaseSampler(SamplerInterface):
    """
    Base sampler class.
    """
    # @TODO: define SAMPLE_FUNCTIONS_DICT automatically:
    # https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-namedefine keywords # noqa
    # noqa
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

    def _check_variables(self):
        self._check_variables_strings()
        self._check_variables_existence()
        self._check_variables_for_dups()

    def _check_variables_strings(self):
        for parameter in self.parameters:
            if not isinstance(parameter, str):
                log_and_raise_exception(
                    "constants and parameters must be strings")

    def _check_variables_existence(self):
        if len(self.parameters) == 0:
            log_and_raise_exception(
                "Either constants or parameters must be included in the " +
                "sampler data")

    def _check_variables_for_dups(self):
        if len(self.parameters) != len(set(self.parameters)):
            dupes = set(find_duplicates(self.parameters))
            log_and_raise_exception(
                "The following constants or parameters are defined more " +
                "than once: " + str(dupes))

    def downselect(self, samples):
        """
        Downselect samples based on specification in sampling_dict.

        Prototype dictionary::

           num_samples: 30
           previous_samples: samples.csv # optional
           parameters:
               X1:
                   min: 10
                   max: 50
               X2:
                   min: 10
                   max: 50
        """
        if not PANDAS_PLUS:
            log_and_raise_exception(
                "This function requires pandas, numpy & scipy packages")

        df = pd.DataFrame.from_dict(self._samples)
        columns = self.parameters
        ndims = len(columns)
        candidates = df[columns].values.tolist()
        num_points = samples

        if not('previous_samples' in self.data.keys()):
            sample_points = []
            sample_points.append(candidates[0])
            new_sample_points = []
            new_sample_points.append(candidates[0])
            new_sample_ids = []
            new_sample_ids.append(0)
            n0 = 1
        else:
            try:
                previous_samples = pd.read_csv(self.data["previous_samples"])
            except ValueError:
                raise Exception("Error opening previous_samples datafile:" +
                                self.data["previous_samples"])
            sample_points = previous_samples[columns].values.tolist()
            new_sample_points = []
            new_sample_ids = []
            n0 = 0

        mins = np.zeros(ndims)
        maxs = np.zeros(ndims)

        first = True
        for i, candidate in enumerate(candidates):
            for j in range(ndims):
                if first:
                    mins[j] = candidate[j]
                    maxs[j] = candidate[j]
                    first = False
                else:
                    mins[j] = min(candidate[j], mins[j])
                    maxs[j] = max(candidate[j], maxs[j])
        print("extrema for new input_labels: ", mins, maxs)
        print("down sampling to %d best candidates from %d total points." % (
            num_points, len(candidates)))
        bign = len(candidates)

        for n in range(n0, num_points):
            px = np.asarray(sample_points)
            tree = spatial.KDTree(px)
            j = bign
            d = 0.0
            for i in range(1, bign):
                pos = candidates[i]
                dist = tree.query(pos)[0]
                if dist > d:
                    j = i
                    d = dist
            if j == bign:
                raise Exception("Something went wrong!")
            else:
                new_sample_points.append(candidates[j])
                sample_points.append(candidates[j])
                new_sample_ids.append(j)

        new_samples_df = pd.DataFrame(columns=df.keys().tolist())
        for new_sample_id in new_sample_ids:
            new_samples_df.append(df.iloc[new_sample_id])

        # for n in range(len(new_sample_ids)):
        #     new_samples_df = new_samples_df.append(df.iloc[new_sample_ids[n]])

        self._samples = new_samples_df.to_dict(orient='records')

    @property
    def parameter_block(self):
        """
        Converts samples to parameter dictionary for ``codepy setup`` and ``codepy run``

        The keys are the labels and the values are a string version of the
        list, so it can be easily passed to Jinja.
        """ # noqa
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
