"""
Module defining the custom sampler object.
"""

import logging

from scisample.random import RandomSampler

LOG = logging.getLogger(__name__)


class BestCandidateSampler(RandomSampler):
    """
    Class defining basic random sampling.

    This is similar to the ``csv`` functionality of ``codepy setup``
    and ``codepy run``.  Its sampler data takes two blocks:
    ``constants`` and ``parameters``:

    .. code:: yaml

        sampler:
            type: random
            num_samples: 30
            previous_samples: samples.csv # optional
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

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        Checks that all entries in ``parameters`` have the same
        length.

        :returns: True if the schema is valid, False otherwise.
        """
        if not super(BestCandidateSampler, self).is_valid():
            return False

        return True

    def get_samples(self, over_sample_rate=10):
        """
        Return set of best candidate samples based
        on specification in sampling_dict.

        Prototype dictionary:

        sample_type: best_candidate
        num_samples: 30
        # previous_samples: samples.csv
        constants:
            X3: 20
        parameters:
            X1:
                min: 10
                max: 50
            X2:
                min: 10
                max: 50
        """
        if self._samples is not None:
            return self._samples

        self._samples = []

        new_sampling_dict = self.data.copy()
        new_sampling_dict["num_samples"] *= over_sample_rate
        new_sampling_dict["type"] = "random"
        new_random_sample = RandomSampler(new_sampling_dict)
        new_random_sample.get_samples()
        new_random_sample.downselect(self.data["num_samples"])

        self._samples = new_random_sample._samples

        return self._samples
