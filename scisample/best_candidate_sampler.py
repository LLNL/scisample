"""
Module defining the best candidate sampler object.
"""

# @TODO: support more parameters than in previous_samples
# @TODO: remove_clustering: find nearest neighbors on all dimensions

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
            previous_samples: samples.csv  # (string) optional
            cost_variable: cost    # (string) required if previous_samples is provided
            cost_target: 1.0       # optional
            cost_target_oversample_ratio: 2.0 # default = BestCandidateSampler.DEFAULT_COST_TARGET_OVERSAMPLE_RATIO
            downselect_ratio: 0.3  # default = BestCandidateSampler.DEFAULT_DOWNSELECT_RATIO
            voxel_overlap: 0.55 # default = BestCandidateSampler.DEFAULT_VOXEL_OVERLAP
            remove_clustering: true # default = BestCandidateSampler.REMOVE_CLUSTERING
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5 max: 10
                X3:
                    min: 5 max: 10

    A total of ``num_samples`` will be generated. Entries in the ``constants``
    dictionary will be added to all samples. Entries in the ``parameters``
    block will be selected from a range of ``min`` to ``max``.  The final
    distribution will be generated using a best candidate algorithm. The result
    of the above block would be something like:

    .. code:: python

        [{X1: 20, X2: 5.632222227306036, X3: 6.633392173916806},
         {X1: 20, X2: 7.44369755967992, X3: 8.941266067294213}]

    If ``previous_samples`` is provided, the algorithm will select points near
    the best points from the previous samples. The ``cost_variable`` is used to
    determine the best points. The ``downselect_ratio`` defines which fraction
    of the previous samples will be used to generate the next set of samples.
    The ``voxel_overlap`` defines overlap between nearest neighbors used for
    sampling. (When the voxel_overlap is 1.0, each voxel surrounding a previous
    point just touches its nearest neighbor.)

    If ``cost_target`` is provided, the algorithm will stop (e.g. return no new
    samples) when the number of samples with a cost less than ``cost_target``
    is greater than or equal to ``num_samples``. The algorithm will also
    include near neighbors less than or equal to ``cost_target_oversample_ratio
    * cost_target`` when selecting the next set of samples. (This helps to
    ensure that the edges valid space are explored.)
    """
    DEFAULT_DOWNSELECT_RATIO = 0.3
    DEFAULT_VOXEL_OVERLAP = 0.55
    DEFAULT_COST_TARGET_OVERSAMPLE_RATIO = 2.0
    DEFAULT_REMOVE_CLUSTERING = True

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
        if "remove_clustering" not in self.data:
            self.data["remove_clustering"] = (
                self.DEFAULT_REMOVE_CLUSTERING)
        if "cost_target_oversample_ratio" not in self.data:
            self.data["cost_target_oversample_ratio"] = (
                self.DEFAULT_COST_TARGET_OVERSAMPLE_RATIO)
        if "downselect_ratio" in self.data:
            if self.data["downselect_ratio"] <= 0.0:
                log_and_raise_exception("The 'downselect_ratio' must be > 0.0.")
            if self.data["downselect_ratio"] > 1.0:
                log_and_raise_exception("The 'downselect_ratio' must be <= 1.0.")
        else:
            self.data["downselect_ratio"] = self.DEFAULT_DOWNSELECT_RATIO
        if "voxel_overlap" in self.data:
            if self.data["voxel_overlap"] <= 0.0:
                log_and_raise_exception("The 'voxel_overlap' must be greater than 0.0.")
        else:
            self.data["voxel_overlap"] = self.DEFAULT_VOXEL_OVERLAP
        if "previous_samples" in self.data:
            required_fields = [
                "cost_variable"]
            missing_fields = []
            for field in required_fields:
                if field not in self.data:
                    missing_fields.append(field)
            if missing_fields:
                log_and_raise_exception(
                    "When using 'previous_samples', best candidate sampling "
                    f"requires '{missing_fields}' fields.")
            else:
                previous_samples = pd.read_csv(Path(self.data["previous_samples"]))
                if self.data["cost_variable"] not in previous_samples.columns:
                    log_and_raise_exception(
                        f"'cost_variable' '{self.data['cost_variable']}' is not present in 'previous_samples'."
                    )

    # @TODO: what is the more correct way to do this?
    # pylint warning
    # W0221 - Parameters differ from overridden 'get_samples' method
    #         (arguments-differ)

    def distance(self, point_1, point_2):
        """ Calculate the distance between two points. """
        return manhattan_distance(point_1, point_2)

    def make_distance_map(self, previous_samples, downselect_parameters):
        distance_map = defaultdict(list)
        min_distance = float("inf")
        max_distance = 0.0
        for i in range(len(previous_samples)):
            point_1 = previous_samples.iloc[i][downselect_parameters]
            for j in range(i+1, len(previous_samples)):
                point_2 = previous_samples.iloc[j][downselect_parameters]
                distance_map[i].append([self.distance(point_1, point_2), j])
                distance_map[j].append([self.distance(point_1, point_2), i])
        for value in distance_map.values():
            value.sort(key=lambda x: x[0])
            min_distance = min(min_distance, value[0][0])
            max_distance = max(max_distance, value[0][0])
        return distance_map, min_distance, max_distance

    def average_rows(self, row1_in, row2_in):
        row1 = row1_in.copy()
        column_names = row1.keys().tolist()
        for column_name in column_names:
            try:
                row1[column_name] = (row1[column_name] + row2_in[column_name]) / 2.0
            except TypeError:
                if not row1[column_name] == row2_in[column_name]:
                    raise SampleError(f"can't average {row1[column_name]} and {row2_in[column_name]}")
        return row1

    def interpolate_points_symmetrically(
            self, previous_samples_in, downselect_parameters):
        # create distance map
        min_distance = 0.0
        max_distance = 1.0
        all_new_rows = []
        previous_samples = previous_samples_in.copy()
        previous_samples = previous_samples.reset_index(drop=True)
        max_count = 2
        count = 0
        while count < max_count and min_distance < max_distance / 2.0:
            count += 1
            print("symmetric count: ", count)
            distance_map, min_distance, max_distance = self.make_distance_map(
                previous_samples, downselect_parameters)
            pair_set = set()
            new_rows = []
            for i, value in distance_map.items():
                j = value[0][1]
                if value[0][0] > 2.0 * min_distance:
                    if (i, j) not in pair_set and (j, i) not in pair_set:
                        pair_set.add((i, j))
                        pair_set.add((j, i))
                        row1 = previous_samples.iloc[i]
                        row2 = previous_samples.iloc[j]
                        new_rows.append(self.average_rows(row1, row2))
            df_new = pd.DataFrame()
            if new_rows:
                all_new_rows.extend(new_rows)
                df_new = pd.DataFrame(all_new_rows)
                previous_samples = pd.concat([previous_samples, df_new], ignore_index=True)
                previous_samples = previous_samples.drop_duplicates()
                previous_samples = previous_samples.reset_index(drop=True)
                distance_map, min_distance, max_distance = self.make_distance_map(
                    previous_samples, downselect_parameters)
        print(f"df_new symmetric: {len(df_new)}; min_distance: {min_distance}; max_distance: {max_distance}")
        return df_new

    def interpolate_points_asymmetrically(
            self, previous_samples_in, downselect_parameters):
        all_new_rows = []
        new_edges = []
        previous_samples = previous_samples_in.copy()
        previous_samples = previous_samples.drop_duplicates()
        previous_samples = previous_samples.reset_index(drop=True)
        interpolated_points = set()
        for sign in [-1, 1]:
            for downselect_parameter in downselect_parameters:
                min_distance = 0.0
                max_distance = 1.0
                max_count = 2
                count = 0
                while count < max_count and min_distance < max_distance / 2.0:
                    count += 1
                    print("asymmetric count: ", count, len(previous_samples), downselect_parameter, sign)
                    min_distance = float("inf")
                    max_distance = 0.0
                    distance_map = defaultdict(list)
                    for i in range(len(previous_samples)):
                        try:
                            point_1 = previous_samples[downselect_parameter].get(i, None)
                            if point_1 is None:
                                print(f"ERROR: Key not found - i={i}, p={downselect_parameter}, len={len(previous_samples)}, {previous_samples.columns}")
                                continue
                        except KeyError as exception:
                            print(
                                f"ERROR: {exception}, i={i}, p={downselect_parameter}, "
                                f"len={len(previous_samples)}, {previous_samples.columns}")
                            # raise
                            continue
                        for j in range(len(previous_samples)):
                            if i == j:
                                continue
                            try:
                                point_2 = previous_samples.loc[j,downselect_parameter]
                            except KeyError as exception:
                                print(
                                    f"ERROR: {exception}, i={i}, p={downselect_parameter}, "
                                    f"len={len(previous_samples)}, {previous_samples.columns}")
                                # raise
                                continue
                            if point_1 * sign > point_2 * sign:
                                distance = float("inf")
                            else:
                                distance = point_2 * sign - point_1 * sign
                            distance_map[i].append([distance, j])
                    for value in distance_map.values():
                        value.sort(key=lambda x: x[0])
                        distance = value[0][0]
                        if distance < float("inf"):
                            min_distance = min(min_distance, distance)
                            max_distance = max(max_distance, distance)
                    _, min_distance2, _ = (
                        self.make_distance_map(
                            previous_samples, downselect_parameters))
                    min_distance = min_distance2
                    new_rows = []
                    for i, value in distance_map.items():
                        distance = value[0][0]
                        j = value[0][1]
                        if distance == float('inf'):
                            row1 = previous_samples.iloc[i].copy()
                            old_value = row1[downselect_parameter]
                            new_value = old_value + sign * min_distance
                            new_value = min(
                                new_value,
                                self.data[
                                    "parameters"][
                                        downselect_parameter]["max"])
                            new_value = max(
                                new_value,
                                self.data[
                                    "parameters"][
                                        downselect_parameter]["min"])
                            row1[downselect_parameter] = new_value
                            if new_value != old_value:
                                new_edges.append(row1)
                        if distance < float('inf') and distance > 2.0 * min_distance:
                            if ((i, j) not in interpolated_points
                                and (j, i) not in interpolated_points):
                                interpolated_points.add((i, j))
                                interpolated_points.add((j, i))
                                row1 = previous_samples.iloc[i]
                                row2 = previous_samples.iloc[j]
                                new_rows.append(self.average_rows(row1, row2))
                    df_new = pd.DataFrame()
                    if new_rows:
                        all_new_rows.extend(new_rows)
                        df_new = pd.DataFrame(all_new_rows)
                        df_new = df_new.drop_duplicates()
                        df_new = df_new.reset_index(drop=True)
                        previous_samples = pd.concat([previous_samples, df_new], ignore_index=True)
                        previous_samples = previous_samples.drop_duplicates()
                        previous_samples = previous_samples.reset_index(drop=True)
        # print(df_new)
        df_edges = pd.DataFrame(new_edges)
        merged = pd.merge(df_edges, previous_samples_in, how='left', indicator=True)
        result = merged[merged['_merge'] == 'left_only']
        result = result.drop('_merge', axis=1)
        df_new = pd.concat([df_new, result], ignore_index=True)
        df_new = df_new.drop_duplicates()
        df_new = df_new.reset_index(drop=True)
        print(f"df_new asymmetric: {len(df_new)}; min_distance: {min_distance}; max_distance: {max_distance}")
        return df_new

    # def new_samples_near_previous_samples(
    #         self, num_samples, previous_samples, downselect_parameters):
    #     """
    #     Given a set of previous samples, generate a new set of samples
    #     near the previous samples.
    #     """
    #     distance_map, min_distance, max_distance = self.make_distance_map(
    #         previous_samples, downselect_parameters)
    #     new_rows = []
    #     for i, value in distance_map.items():
    #         j = value[0][1]
    #         new_row = {}
    #         for parameter in downselect_parameters:
    #             new_row[parameter] = (
    #                 previous_samples.iloc[i][parameter]
    #                 + previous_samples.iloc[j][parameter]) / 2.0
    #         new_rows.append(new_row)
    #     return new_rows

    def add_missing_data_to_df(self, df):
        df_dict_list = df.to_dict(orient='records')
        df_dict_list = [self.add_missing_data(row) for row in df_dict_list]
        return pd.DataFrame(df_dict_list)

    def add_missing_data(self, new_row):
        new_sampling_dict = copy.deepcopy(self.data)
        new_sampling_dict["type"] = "random"
        new_sampling_dict["num_samples"] = 1
        sampler = RandomSampler(new_sampling_dict)
        new_sample = sampler.get_samples()[0]
        for key, value in new_row.items():
            new_sample[key] = value
        return new_sample

    def get_nearby_candidates(
            self, combined_samples, downselect_parameters, over_sample_rate=10):
        """
        Get nearby candidates.

        """
        if self.data["over_sample_rate"] is not None:
            over_sample_rate = self.data["over_sample_rate"]

        distance_map, _, _ = self.make_distance_map(
            combined_samples, downselect_parameters)

        num_samples = self.data["num_samples"]
        num_samples *= over_sample_rate * 1.0
        num_samples /= len(combined_samples)
        num_samples = int(num_samples + 0.5)
        sampler_list = []
        for i, value in distance_map.items():
            j = value[0][1]
            new_sampling_dict = copy.deepcopy(self.data)
            for parameter in downselect_parameters:
                half_width = 0.5 * abs(
                    combined_samples.iloc[i][parameter]
                    - combined_samples.iloc[j][parameter])
                new_sampling_dict["parameters"][parameter]["min"] = (
                    combined_samples.iloc[i][parameter] - half_width)
                new_sampling_dict["parameters"][parameter]["max"] = (
                    combined_samples.iloc[i][parameter] + half_width)
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
        return pd.DataFrame(new_samples)

    # def get_samples_with_previous_old(self, over_sample_rate=10):
    #     """
    #     Get samples from the sampler.

    #     This returns samples as a list of dictionaries, with the
    #     sample variables as the keys:

    #     .. code:: python

    #         [{'b': 0.89856, 'a': 1}, {'b': 0.923223, 'a': 1}, ... ]
    #     """
    #     if self._samples is not None:
    #         return self._samples

    #     if self.data["over_sample_rate"] is not None:
    #         over_sample_rate = self.data["over_sample_rate"]

    #     self._samples = []
    #     previous_samples = pd.read_csv(Path(self.data["previous_samples"]))
    #     num_samples_less_than_overtarget = 0
    #     if "cost_target" in self.data:
    #         num_samples_less_than_target = len(
    #             previous_samples[
    #                 previous_samples[self.data["cost_variable"]]
    #                 < self.data["cost_target"]])
    #         if num_samples_less_than_target >= self.data["num_samples"]:
    #             self._samples = []
    #             return self._samples
    #         else:
    #             num_samples_less_than_overtarget = len(
    #                 previous_samples[
    #                     previous_samples[self.data["cost_variable"]]
    #                     < self.data["cost_target"]
    #                     * self.data["cost_target_oversample_ratio"]])
    #     previous_headers = list(previous_samples.columns)
    #     # sort previous samples by cost
    #     previous_samples = previous_samples.sort_values(
    #         by=self.data["cost_variable"])
    #     LOG.warning(f"previous_samples: {previous_samples[self.data['cost_variable']][:5].to_list()}")
    #     num_samples_to_keep = int(
    #         self.data["num_samples"] * self.data["downselect_ratio"])
    #     num_samples_to_keep = max(num_samples_to_keep, num_samples_less_than_overtarget)
    #     if num_samples_to_keep > self.data["num_samples"]:
    #         LOG.warning("The number of samples to keep is greater than the "
    #                     "number of samples to generate. The number of samples "
    #                     "to generate will be increased to the number of "
    #                     "samples to keep.")
    #         self.data["num_samples"] = num_samples_to_keep
    #     previous_samples = previous_samples[:num_samples_to_keep]
    #     downselect_parameters = [
    #         parameter
    #         for parameter in self.data["parameters"].keys()
    #         if parameter in previous_headers]
    #     if not downselect_parameters:
    #         return self.get_samples_no_previous(over_sample_rate)
    #     new_rows = self.interpolate_points_symmetrically(
    #         previous_samples, downselect_parameters)
    #     if new_rows:
    #         print(f"symmetrically interpolated {len(new_rows)} points")
    #         df_new = pd.DataFrame(new_rows)
    #         # previous_samples = previous_samples.append(new_rows, ignore_index=True)
    #         previous_samples = pd.concat([previous_samples, df_new], ignore_index=True)
    #     new_rows = self.interpolate_points_asymmetrically(
    #         previous_samples, downselect_parameters)
    #     if new_rows:
    #         print(f"<assymmetrically interpolated {len(new_rows)} points")
    #         print(f"new_rows: {new_rows}")
    #         df_new = pd.DataFrame(new_rows)
    #         # previous_samples = previous_samples.append(new_rows, ignore_index=True)
    #         previous_samples = pd.concat([previous_samples, df_new], ignore_index=True)

    #     distance_map, min_distance, max_distance = self.make_distance_map(
    #         previous_samples, downselect_parameters)
    #     defaultdict(list)

    #     num_samples = self.data["num_samples"]
    #     num_samples *= over_sample_rate * 1.0
    #     num_samples /= num_samples_to_keep
    #     num_samples = int(num_samples + 0.5)
    #     sampler_list = []
    #     for i, value in distance_map.items():
    #         j = value[0][1]
    #         new_sampling_dict = copy.deepcopy(self.data)
    #         for parameter in downselect_parameters:
    #             half_width = self.data["voxel_overlap"] * abs(
    #                 previous_samples.iloc[i][parameter]
    #                 - previous_samples.iloc[j][parameter])
    #             new_sampling_dict["parameters"][parameter]["min"] = (
    #                 previous_samples.iloc[i][parameter] - half_width)
    #             new_sampling_dict["parameters"][parameter]["max"] = (
    #                 previous_samples.iloc[i][parameter] + half_width)
    #             if (new_sampling_dict["parameters"][parameter]["min"]
    #                     < self.data["parameters"][parameter]["min"]):
    #                 new_sampling_dict["parameters"][parameter]["min"] = (
    #                     self.data["parameters"][parameter]["min"])
    #             if (new_sampling_dict["parameters"][parameter]["max"]
    #                     > self.data["parameters"][parameter]["max"]):
    #                 new_sampling_dict["parameters"][parameter]["max"] = (
    #                     self.data["parameters"][parameter]["max"])
    #         new_sampling_dict["num_samples"] = num_samples
    #         sampler_list.append(RandomSampler(new_sampling_dict))
    #     new_samples = []
    #     for sampler in sampler_list:
    #         new_samples.extend(sampler.get_samples())
    #     self._samples = []
    #     for sample in new_samples:
    #         _sample = {}
    #         for parameter in downselect_parameters:
    #             _sample[parameter] = sample[parameter]
    #         self._samples.append(_sample)
    #     try:
    #         new_indices = self.downselect(
    #             self.data["num_samples"],
    #             previous_samples=previous_samples,
    #             return_indices=True)
    #     except Exception as exception:  # pylint: disable=broad-except
    #         log_and_raise_exception(
    #             f"Error during 'downselect' in 'best_candidate' "
    #             f"sampling: {exception}")
    #     self._samples = [new_samples[i] for i in new_indices]
    #     return self._samples

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
        previous_samples = self.add_missing_data_to_df(previous_samples)

        num_samples_less_than_overtarget = 0
        if "cost_target" in self.data:
            num_samples_less_than_target = len(
                previous_samples[
                    previous_samples[self.data["cost_variable"]]
                    < self.data["cost_target"]])
            if num_samples_less_than_target >= self.data["num_samples"]:
                self._samples = []
                return self._samples
            else:
                num_samples_less_than_overtarget = len(
                    previous_samples[
                        previous_samples[self.data["cost_variable"]]
                        < self.data["cost_target"]
                        * self.data["cost_target_oversample_ratio"]])

        # sort previous samples by cost
        previous_samples = previous_samples.sort_values(
            by=self.data["cost_variable"])
        num_samples_to_keep = int(
            self.data["num_samples"] * self.data["downselect_ratio"])
        num_samples_to_keep = max(num_samples_to_keep, num_samples_less_than_overtarget)
        previous_samples = previous_samples[:num_samples_to_keep]

        previous_headers = list(previous_samples.columns)
        downselect_parameters = [
            parameter
            for parameter in self.data["parameters"].keys()
            if parameter in previous_headers]
        if not downselect_parameters:
            return self.get_samples_no_previous(over_sample_rate)
        if self.data["remove_clustering"]:
            new_rows_1 = self.interpolate_points_asymmetrically(
                previous_samples, downselect_parameters)
            new_rows_2 = self.interpolate_points_symmetrically(
                previous_samples, downselect_parameters)
            new_rows = pd.concat([new_rows_1, new_rows_2], ignore_index=True)
            new_rows = new_rows.drop_duplicates()
            combined_samples = pd.concat(
                [previous_samples, new_rows], ignore_index=True)
            combined_samples = combined_samples.drop_duplicates()
            combined_samples = combined_samples.reset_index(drop=True)
        else:
            combined_samples = previous_samples
        new_candidates = self.get_nearby_candidates(
            combined_samples, downselect_parameters, over_sample_rate)
        if self.data["remove_clustering"]:
            new_candidates = pd.concat(
                [new_candidates, new_rows], ignore_index=True)
        new_candidates = new_candidates.drop_duplicates()
        new_candidates = new_candidates.reset_index(drop=True)
        self._samples = []
        new_candidates_dict_list = new_candidates.to_dict(orient='records')
        for sample in new_candidates_dict_list:
            _sample = {}
            for parameter in downselect_parameters:
                _sample[parameter] = sample[parameter]
            _sample = self.add_missing_data(_sample)
            self._samples.append(_sample)
        try:
            new_indices = self.downselect(
                self.data["num_samples"],
                previous_samples=previous_samples,
                downselect_parameters=downselect_parameters,
                return_indices=True)
        except Exception as exception:  # pylint: disable=broad-except
            log_and_raise_exception(
                f"Error during 'downselect' in 'best_candidate' "
                f"sampling: {exception}")
        self._samples = [new_candidates_dict_list[i] for i in new_indices]
        new_rows_dict_list = new_rows.to_dict(orient='records')
        count = 0
        for new_row in new_rows_dict_list:
            if new_row not in self._samples:
                count += 1
                self._samples.append(new_row)
        if count:
            print(f"added {count} new rows from interpolation")
        # self._samples = [
        #     self.add_missing_data(_sample)
        #     for _sample in self._samples]
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
        # new_sampling_dict["type"] = "random"
        new_sampling_dict["type"] = "best_candidate"
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