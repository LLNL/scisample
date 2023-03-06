"""
Module defining the custom sampler object.
"""

# @TODO: support more parameters than in previous_samples

import logging
import os
import copy
from pathlib import Path
import pandas as pd
from collections import defaultdict

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

    def distance(self, a, b):
        return manhattan_distance(a, b)

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
        previous_samples = pd.read_csv(Path(self.data["previous_samples"]))
        previous_headers = list(previous_samples.columns)

        # sort previous samples by cost
        previous_samples = previous_samples.sort_values(
            by=self.data["cost_variable"])

        # downselect previous samples
        num_previous_samples = len(previous_samples)
        num_samples_to_keep = int(num_previous_samples * self.data["downselect_ratio"])
        previous_samples = previous_samples[:num_samples_to_keep]
        # print(self.data["parameters"])
        downselect_parameters = [
            parameter
            for parameter in self.data["parameters"].keys()
            if parameter in previous_headers]
        if not downselect_parameters:
            return self.get_samples_no_previous(over_sample_rate)
        previous_samples_inputs = previous_samples[downselect_parameters]
        # create distance map
        distance_map = defaultdict(list)
        for i in range(len(previous_samples)):
            a = previous_samples.iloc[i][downselect_parameters]
            for j in range(i+1, len(previous_samples)):
                b = previous_samples.iloc[j][downselect_parameters]
                distance_map[i].append([self.distance(a,b),j])
                distance_map[j].append([self.distance(a,b),i])
        for value in distance_map.values():
            value.sort(key=lambda x: x[0])
        num_samples = self.data["num_samples"]
        num_samples *= over_sample_rate * 1.0
        num_samples /= num_samples_to_keep
        num_samples = int(num_samples + 0.5)
        sampler_list = []
        print(distance_map[5][:7])
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
        # create dataframe from list of dicts
        for sample in new_samples[:5]:
            print("sample:", sample)
            print('sample["X1"]', sample["X1"])
        self._samples = new_samples
        # try:
        if True:
            self.downselect(
                self.data["num_samples"],
                previous_samples=previous_samples)
        # except Exception as exception:
        #     log_and_raise_exception(
        #         f"Error during 'downselect' in 'best_candidate' "
        #         f"sampling: {exception}")

        def rosenbrock(x, y):
            return (1 - x)**2 + 100*(y - x**2)**2
        df = pd.DataFrame(self._samples)
        df['cost'] = df.apply(lambda row: rosenbrock(row['X1'], row['X2']), axis=1)
        df.to_csv("best_candidate_2.csv")
        # raise Exception("stop")

        return self._samples

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

        return self.get_samples_with_previous(over_sample_rate)
