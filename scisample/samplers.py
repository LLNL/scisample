"""
Module defining different sampler interfaces.
"""

import logging

from scisample.base_sampler import BaseSampler
from scisample.best_candidate_sampler import BestCandidateSampler
from scisample.column_list_sampler import ColumnListSampler
from scisample.cross_product_sampler import CrossProductSampler
from scisample.csv_sampler import CsvSampler
from scisample.custom_sampler import CustomSampler
from scisample.list_sampler import ListSampler
from scisample.random_sampler import RandomSampler
from scisample.uqpipeline_sampler import UQPipelineSampler
from scisample.utils import SamplingError, log_and_raise_exception

LOG = logging.getLogger(__name__)

BaseSampler.SAMPLE_FUNCTIONS_DICT = {
    'list': ListSampler,
    'best_candidate': BestCandidateSampler,
    'column_list': ColumnListSampler,
    'cross_product': CrossProductSampler,
    'csv': CsvSampler,
    'random': RandomSampler,
    'custom': CustomSampler,
    'uqpipeline': UQPipelineSampler,
}

BaseSampler.SAMPLE_FUNCTIONS_KEYS = BaseSampler.SAMPLE_FUNCTIONS_DICT.keys()


def new_sampler(sampler_data):
    """
    Dispatch the sampler for the requested sampler data.

    If there is no ``type`` entry in the data, it will raise a
    ``SamplingError``.

    If the ``type`` entry does not match one of the built-in
    samplers, it will raise a ``SamplingError``. Currently the built-in
    samplers are:

    | * ``best_candidate``
    | * ``column_list``
    | * ``cross_product``
    | * ``csv``
    | * ``custom``
    | * ``list``
    | * ``random``

    :param sampler_data: data to validate.
    :returns: Sampler object matching the data.
    """

    if 'type' not in sampler_data:
        log_and_raise_exception(
            f"No type entry in sampler data {sampler_data}")

    try:
        sampler = BaseSampler.SAMPLE_FUNCTIONS_DICT[sampler_data['type']]
    except KeyError:
        log_and_raise_exception(
            f"{sampler_data['type']} " +
            "is not a recognized sampler type")

    try:
        return sampler(sampler_data)
    except SamplingError as exception:
        log_and_raise_exception(exception)
