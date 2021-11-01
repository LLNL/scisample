"""
Module defining the random sampler object.
"""

import logging
import random
from contextlib import suppress

from scisample.base_sampler import BaseSampler
from scisample.utils import log_and_raise_exception, test_for_min_max

LOG = logging.getLogger(__name__)


class RandomSampler(BaseSampler):
    """
    Class defining basic random sampling.

    .. code:: yaml

        sampler:
            type: random
            num_samples: 5
            previous_samples: samples.csv # not supported yet
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5
                    max: 10
                X3:
                    min: 5
                    max: 10

    A total of ``num_samples`` will be generated. Entries in the ``constants``
    dictionary will be added to all samples. Entries in the ``parameters``
    block will be selected from a range of ``min`` to ``max``.  The result of
    the above block would something like:

    .. code:: python

        [{X1: 20, X2: 5.632222227306036, X3: 6.633392173916806},
         {X1: 20, X2: 7.44369755967992, X3: 8.941266067294213}]
    """

    def __init__(self, data):
        """
        Initialize the sampler.

        :param data: Dictionary of sampler data.
        """
        super().__init__(data)
        self.check_validity()

    def check_validity(self):
        super().check_validity()
        self._check_variables()

        # @TODO: test that file exists and it contains the right parameters
        if 'previous_samples' in self.data.keys():
            log_and_raise_exception(
                "'previous_samples' is not yet supported.\n"
                "  Please contact Chris Krenn or Brian Daub for assistance.")

        # @TODO: add error check to schema
        test_for_min_max(self.data["parameters"])

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        return self._parameters_constants_parameters_only()

    def get_samples(self):
        """
        Get samples from the sampler.

        This returns samples as a list of dictionaries, with the
        sample variables as the keys:

        .. code:: python

            [{'b': 0.89856, 'a': 1}, {'b': 0.923223, 'a': 1}, ... ]
        """
        if self._samples is not None:
            return self._samples

        self._samples = []

        random_list = []
        min_dict = {}
        range_dict = {}
        box = []

        for key, value in self.data["parameters"].items():
            min_dict[key] = value["min"]
            range_dict[key] = value["max"] - value["min"]
            box.append([value["min"], value["max"]])

        for i in range(self.data["num_samples"]):
            random_dictionary = {}
            for key, value in self.data["parameters"].items():
                random_dictionary[key] = (
                    min_dict[key] + random.random() * range_dict[key])
            random_list.append(random_dictionary)

        for i in range(len(random_list)):
            new_sample = {}

            with suppress(KeyError):
                new_sample.update(self.data['constants'])

            with suppress(KeyError):
                for key, value in random_list[i].items():
                    new_sample[key] = value

            self._samples.append(new_sample)

        return self._samples