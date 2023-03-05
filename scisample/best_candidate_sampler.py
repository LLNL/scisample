"""
Module defining the custom sampler object.
"""

import logging

from scisample.random_sampler import RandomSampler
from scisample.utils import (log_and_raise_exception, read_csv,
                             manhattan_distance)

LOG = logging.getLogger(__name__)


class BestCandidateSampler(RandomSampler):
    """
    Class defining best candidate sampling.

    .. code:: yaml

        sampler:
            type: best_candidate
            num_samples: 30
            previous_samples: samples.csv # optional
            cost_variable: cost   # required if previous_samples is provided
            downselect_ratio: 0.3 # required if previous_samples is provided
            voxel_overlap: 1.0    # required if previous_samples is provided
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
    block will be selected from a range of ``min`` to ``max``.  The final
    distribution will be generated using a best candidate algorithm. The
    result of the above block would be something like:

    .. code:: python

        [{X1: 20, X2: 5.632222227306036, X3: 6.633392173916806},
         {X1: 20, X2: 7.44369755967992, X3: 8.941266067294213}]

    If ``previous_samples`` is provided, the algorithm will select points
    near the best points from the previous samples. The ``cost_variable``
    is used to determine the best points. The ``downselect_ratio`` defines
    which fraction of the previous samples will be used to generate the
    next set of samples. The ``voxel_overlap`` defines overlap between nearest
    neighbors used for sampling. (When the voxel_overlap is 1.0, each voxel
    surrounding a previous point just touches its nearest neighbor.)
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
        if "downselect_ratio" in self.data:
            if self.data["downselect_ratio"] <= 0.0:
                log_and_raise_exception(
                    "The 'downselect_ratio' must be greater than 0.0.")
            if self.data["downselect_ratio"] > 1.0:
                log_and_raise_exception(
                    "The 'downselect_ratio' must be less than or equal to 1.0.")
        if "voxel_overlap" in self.data:
            if self.data["voxel_overlap"] <= 0.0:
                log_and_raise_exception(
                    "The 'voxel_overlap' must be greater than 0.0.")
        if "previous_samples" in self.data:
            required_fields = ["cost_variable", "downselect_ratio", "voxel_overlap"]
            missing_fields = []
            for field in required_fields:
                if field not in self.data:
                    missing_fields.append(field)
            if missing_fields:
                log_and_raise_exception(
                    "When using 'previous_samples', best candidate "
                    f"sampling requires '{missing_fields}' fields.")

    # @TODO: add more error checking
    # right now, error checking for RandomSampler is sufficient
    # def check_validity(self):
    #     pass

    # @TODO: what is the more correct way to do this?
    # pylint: warning
    # W0221 - Parameters differ from overridden 'get_samples' method
    #         (arguments-differ)

    def get_samples_with_previous(self, over_sample_rate=10):
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

        # read in previous samples
        previous_samples = read_csv(self.data["previous_samples"])
        if not previous_samples:
            log_and_raise_exception(
                "No samples found in 'previous_samples' file.")
        if self.data["cost_variable"] not in previous_samples[0]:
            log_and_raise_exception(
                f"Cost variable '{self.data['cost_variable']}' not found "
                "in 'previous_samples' file.")

        # sort previous samples by cost
        previous_samples.sort(
            key=lambda sample: sample[self.data["cost_variable"]])

        # downselect previous samples
        num_previous_samples = len(previous_samples)
        num_samples_to_keep = int(num_previous_samples * self.data["downselect_ratio"])
        previous_samples = previous_samples[:num_samples_to_keep]

        # #
        # distance_map = {}


        # # get the voxel size
        # voxel_size = self._get_voxel_size(previous_samples)

        # # get the voxel grid
        # voxel_grid = self._get_voxel_grid(previous_samples, voxel_size)

        # # get the samples
        # self._samples = self._get_samples(
        #     previous_samples, voxel_grid, voxel_size, over_sample_rate)

        # return self._samples

    def get_samples_no_previous(self, over_sample_rate=10):
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

        new_sampling_dict = {}
        records_to_copy = ["num_samples", "constants", "parameters"]
        for record in records_to_copy:
            new_sampling_dict[record] = self.data[record]
        new_sampling_dict["num_samples"] *= over_sample_rate
        new_sampling_dict["type"] = "random"
        new_random_sample = RandomSampler(new_sampling_dict)
        new_random_sample.get_samples()
        try:
            new_random_sample.downselect(self.data["num_samples"])
        except Exception as exception:
            log_and_raise_exception(
                f"Error during 'downselect' in 'best_candidate' "
                f"sampling: {exception}")
        self._samples = new_random_sample._samples

        return self._samples

    def get_samples(self, over_sample_rate=10):
        """
        Get samples from the sampler.

        This returns samples as a list of dictionaries, with the
        sample variables as the keys:

        .. code:: python

            [{'b': 0.89856, 'a': 1}, {'b': 0.923223, 'a': 1}, ... ]
        """
        if self._samples is not None:
            return self._samples

        if not "previous_samples" in self.data:
            return self.get_samples_no_previous(over_sample_rate)
        self._samples = []

        return None