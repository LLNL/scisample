"""
Module defining the custom sampler object.
"""

# @TODO: support more parameters than in previous_samples

import logging
import copy
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scisample.random_sampler import RandomSampler
from scisample.utils import (log_and_raise_exception, manhattan_distance)
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
            downselect_ratio: 0.3 # downselect_ratio or downselect_number is required if previous_samples is provided
            downselect_number: 10 # downselect_ratio or downselect_number is required if previous_samples is provided
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
        if "over_sample_rate" not in self.data:
            self.data["over_sample_rate"] = None

    def check_validity(self):
        super().check_validity()
        self._check_variables()
        if "downselect_ratio" in self.data:
            if self.data["downselect_ratio"] <= 0.0:
                log_and_raise_exception(
                    "The 'downselect_ratio' must be > 0.0.")
            if self.data["downselect_ratio"] > 1.0:
                log_and_raise_exception(
                    "The 'downselect_ratio' must be <= 1.0.")
        if "voxel_overlap" in self.data:
            if self.data["voxel_overlap"] <= 0.0:
                log_and_raise_exception(
                    "The 'voxel_overlap' must be greater than 0.0.")
        if "previous_samples" in self.data:
            required_fields = [
                "cost_variable", "downselect_ratio", "voxel_overlap"]
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
    # pylint warning
    # W0221 - Parameters differ from overridden 'get_samples' method
    #         (arguments-differ)

    def distance(self, point_1, point_2):
        """ Calculate the distance between two points. """
        return manhattan_distance(point_1, point_2)

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

        if self.data["over_sample_rate"] is not None:
            over_sample_rate = self.data["over_sample_rate"]

        self._samples = []
        previous_samples = pd.read_csv(Path(self.data["previous_samples"]))
        previous_headers = list(previous_samples.columns)

        # sort previous samples by cost
        previous_samples = previous_samples.sort_values(
            by=self.data["cost_variable"])
        LOG.warning(f"previous_samples: {previous_samples[self.data['cost_variable']][:5].to_list()}")
        # downselect previous samples
        num_previous_samples = len(previous_samples)
        num_samples_to_keep = int(
            self.data["num_samples"] * self.data["downselect_ratio"])
        # num_samples_to_keep = min(num_samples_to_keep, self.data["downselect_number"])
        if num_samples_to_keep > self.data["num_samples"]:
            LOG.warning("The number of samples to keep is greater than the "
                        "number of samples to generate. The number of samples "
                        "to generate will be increased to the number of "
                        "samples to keep.")
            self.data["num_samples"] = num_samples_to_keep
        previous_samples = previous_samples[:num_samples_to_keep]
        downselect_parameters = [
            parameter
            for parameter in self.data["parameters"].keys()
            if parameter in previous_headers]
        if not downselect_parameters:
            return self.get_samples_no_previous(over_sample_rate)
        # create distance map
        distance_map = defaultdict(list)
        for i in range(len(previous_samples)):
            point_1 = previous_samples.iloc[i][downselect_parameters]
            for j in range(i+1, len(previous_samples)):
                point_2 = previous_samples.iloc[j][downselect_parameters]
                distance_map[i].append([self.distance(point_1, point_2), j])
                distance_map[j].append([self.distance(point_1, point_2), i])
        for value in distance_map.values():
            value.sort(key=lambda x: x[0])
        num_samples = self.data["num_samples"]
        num_samples *= over_sample_rate * 1.0
        num_samples /= num_samples_to_keep
        num_samples = int(num_samples + 0.5)
        sampler_list = []
        for i, value in distance_map.items():
            j = value[0][1]
            new_sampling_dict = copy.deepcopy(self.data)
            for parameter in downselect_parameters:
                half_width = self.data["voxel_overlap"] * abs(
                    previous_samples.iloc[i][parameter]
                    - previous_samples.iloc[j][parameter])
                new_sampling_dict["parameters"][parameter]["min"] = (
                    previous_samples.iloc[i][parameter] - half_width)
                new_sampling_dict["parameters"][parameter]["max"] = (
                    previous_samples.iloc[i][parameter] + half_width)
                if (new_sampling_dict["parameters"][parameter]["min"]
                        < self.data["parameters"][parameter]["min"]):
                    new_sampling_dict["parameters"][parameter]["min"] = (
                        self.data["parameters"][parameter]["min"])
                if (new_sampling_dict["parameters"][parameter]["max"]
                        > self.data["parameters"][parameter]["max"]):
                    new_sampling_dict["parameters"][parameter]["max"] = (
                        self.data["parameters"][parameter]["max"])
            new_sampling_dict["num_samples"] = num_samples
            sampler_list.append(RandomSampler(new_sampling_dict))
        new_samples = []
        for sampler in sampler_list:
            new_samples.extend(sampler.get_samples())
        self._samples = []
        for sample in new_samples:
            _sample = {}
            for parameter in downselect_parameters:
                _sample[parameter] = sample[parameter]
            self._samples.append(_sample)
        try:
            new_indices = self.downselect(
                self.data["num_samples"],
                previous_samples=previous_samples,
                return_indices=True)
        except Exception as exception:  # pylint: disable=broad-except
            log_and_raise_exception(
                f"Error during 'downselect' in 'best_candidate' "
                f"sampling: {exception}")
        self._samples = [new_samples[i] for i in new_indices]
        return self._samples

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

        if self.data["over_sample_rate"] is not None:
            over_sample_rate = self.data["over_sample_rate"]

        self._samples = []

        new_sampling_dict = {}
        records_to_copy = ["num_samples", "constants", "parameters"]
        for record in records_to_copy:
            if record in self.data:
                new_sampling_dict[record] = self.data[record]
        new_sampling_dict["num_samples"] *= over_sample_rate
        new_sampling_dict["type"] = "random"
        new_random_sample = RandomSampler(new_sampling_dict)
        new_random_sample.get_samples()
        try:
            new_random_sample.downselect(self.data["num_samples"])
        except Exception as exception:  # pylint: disable=broad-except
            log_and_raise_exception(
                f"Error during 'downselect' in 'best_candidate' "
                f"sampling: {exception}")
        self._samples = new_random_sample.get_samples()

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

        if "previous_samples" not in self.data:
            return self.get_samples_no_previous(over_sample_rate)

        return self.get_samples_with_previous(over_sample_rate)
