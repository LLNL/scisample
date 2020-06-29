"""
Module defining the custom sampler object.
"""

import logging
import random

from contextlib import suppress
from scisample.base_sampler import BaseSampler

PANDAS_PLUS = False
with suppress(ModuleNotFoundError):
    import pandas as pd
    import numpy as np
    import scipy.spatial as spatial
    PANDAS_PLUS = True

LOG = logging.getLogger(__name__)

# def best_candidate_sample(sampling_dict, over_sample_rate=10):
#     """
#     Return set of best candidate samples based
#     on specification in sampling_dict.

#     Prototype dictionary:

#     sample_type: best_candidate
#     num_samples: 30
#     # previous_samples: samples.csv
#     constants:
#         X3: 20
#     parameters:
#         X1:
#             min: 10
#             max: 50
#         X2:
#             min: 10
#             max: 50
#     """
#     _log_assert(
#         PANDAS_PLUS,
#         "This function requires pandas, numpy, scipy & sklearn packages")
#     _validate_best_candidate_dictionary(sampling_dict)
#     new_sampling_dict = sampling_dict.copy()
#     new_sampling_dict["num_samples"] *= over_sample_rate
#     new_random_sample = random_sample(new_sampling_dict)

#     samples = downselect(new_random_sample, sampling_dict)
#     if "constants" in sampling_dict.keys():
#         for sample in samples:
#             for key, value in sampling_dict["constants"].items():
#                 sample[key] = value

#     return samples
