"""
Module defining different sampler objects.
"""

import logging
from contextlib import suppress
import itertools
from pathlib import Path
import importlib
import sys

from jsonschema import ValidationError

from scisample.interface import SamplerInterface
from scisample.schema import validate_sampler
from scisample.utils import (
    read_csv, transpose_tabular, list_to_csv, _convert_dict_to_maestro_params,
    find_duplicates)

MAESTROWF = False
with suppress(ModuleNotFoundError):
    from maestrowf.datastructures.core import ParameterGenerator
    MAESTROWF = True

LOG = logging.getLogger(__name__)


def new_sampler(sampler_data):
    """
    Dispatch the sampler for the requested sampler data.

    If there is no ``type`` entry in the data, it will raise a
    ``ValueError``.

    If the ``type`` entry does not match one of the built-in
    samplers, it will raise a ``KeyError``.  Currently the
    three built in samplers are ``custom``, ``cross_product``,
    ``list``, and ``csv``.

    :param sampler_data: data to validate.
    :returns: Sampler object matching the data.
    """

    # SAMPLE_FUNCTIONS_DICT is defined below class definitions

    if 'type' not in sampler_data:
        raise ValueError(f"No type entry in sampler data {sampler_data}")

    try:
        sampler = SAMPLE_FUNCTIONS_DICT[sampler_data['type']]
    except KeyError:
        raise KeyError(f"{sampler_data['type']} is not a recognized sampler type")

    return sampler(sampler_data)

    # if sampler_data['type'] == 'list':
    #     return ListSampler(sampler_data)

    # if sampler_data['type'] == 'cross_product':
    #     return CrossProductSampler(sampler_data)

    # if sampler_data['type'] == 'csv':
    #     return CsvSampler(sampler_data)

    # if sampler_data['type'] == 'custom':
    #     return CustomSampler(sampler_data)


class BaseSampler(SamplerInterface):
    """
    Base sampler class.
    """

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


class ListSampler(BaseSampler):
    """
    Class defining basic list sampling.

    This is similar to the ``csv`` functionality of ``codepy setup``
    and ``codepy run``.  Its sampler data takes two blocks:
    ``constants`` and ``parameters``:

    .. code:: yaml

        sampler:
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]

    Entries in the ``constants`` dictionary will be added to all samples.
    Entries in the ``parameters`` block will be matched one to one.  The
    result of the above block would be:

    .. code:: python

        [{X1: 20, X2: 5, X3: 5}, {X1: 20, X2: 10, X3: 10}]
    """

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        Checks that all entries in ``parameters`` have the same
        length.

        :returns: True if the schema is valid, False otherwise.
        """
        LOG.info("ListSampler.is_valid()")
        if not super(ListSampler, self).is_valid():
            return False

        test_length = None

        with suppress(KeyError):
            for key, value in self.data['parameters'].items():
                if test_length is None:
                    test_length = len(value)
                if len(value) != test_length:
                    LOG.error(
                        "All parameters must have the same nuumber of entries"
                        )
                    return False

        if 'constants' not in self.data and 'parameters' not in self.data:
            LOG.error(
                "Either constants or parameters must be included in the "
                "sampler data"
                )
            return False
        LOG.info("testing for duplicates")
        if len(self.parameters) != len(set(self.parameters)):
            dupes = set(find_duplicates(self.parameters))
            LOG.error(
                "The following constants or parameters are defined more"
                "than once: " + str(dupes)
                )
            return False

        return True

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        parameters = []
        with suppress(KeyError):
            parameters.extend(list(self.data['constants'].keys()))
        with suppress(KeyError):
            parameters.extend(list(self.data['parameters'].keys()))

        return parameters

    def get_samples(self):
        """
        Get samples from the sampler.
 
        This returns samples as a list of dictionaries, with the
        sample variables as the keys:

        .. code:: python

            [{'b': 0.89856, 'a': 1}, {'b': 0.923223, 'a': 1}, ... ]
        """
        LOG.info("ListSampler.get_samples()")
        if self._samples is not None:
            return self._samples

        self._samples = []

        num_samples = 1

        with suppress(KeyError):
            for key, value in self.data['parameters'].items():
                num_samples = len(value)
                break

        for i in range(num_samples):
            new_sample = {}

            with suppress(KeyError):
                new_sample.update(self.data['constants'])

            with suppress(KeyError):
                for key, value in self.data['parameters'].items():
                    new_sample[key] = value[i]

            self._samples.append(new_sample)

        return self._samples


class ColumnListSampler(BaseSampler):
    """
    Class defining basic column list sampling.

    This is similar to the ``csv`` functionality of ``codepy setup``
    and ``codepy run``.  Its sampler data takes two blocks:
    ``constants`` and ``parameters``:

    .. code:: yaml

        sampler:
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2       X3
                5        5
                10       10

    Entries in the ``constants`` dictionary will be added to all samples.
    Entries in the ``parameters`` block will be matched one to one.  The
    result of the above block would be:

    .. code:: python

        [{X1: 20, X2: 5, X3: 5}, {X1: 20, X2: 10, X3: 10}]
    """

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        Checks that all entries in ``parameters`` have the same
        length.

        :returns: True if the schema is valid, False otherwise.
        """
        if not super(ColumnListSampler, self).is_valid():
            return False

        test_length = None

        if 'constants' not in self.data and 'parameters' not in self.data:
            LOG.error(
                "Either constants or parameters must be included in the "
                "sampler data"
                )
            return False
        return True

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        parameters = []
        with suppress(KeyError):
            parameters.extend(list(self.data['constants'].keys()))
        with suppress(KeyError):
            rows = self.data['parameters'].split('\n')
            headers = rows.pop(0).split()
            parameters.extend(headers)
        return parameters

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

        num_samples = len(self.parameters)

        parameter_samples = []
        with suppress(KeyError):
            rows = self.data['parameters'].split('\n')
            headers = rows.pop(0).split()
            for row in rows:
                data = row.split()
                if len(data) > 0:
                    if len(data) != len(headers):
                        LOG.error(
                            "All parameters must have the same nuumber of entries"
                        )
                        return False
                    sample = {}
                    for header, datum in zip(headers, data):
                        sample[header] = datum
                    parameter_samples.append(sample)

        for i in range(num_samples):
            new_sample = {}

            with suppress(KeyError):
                new_sample.update(self.data['constants'])

            with suppress(KeyError):
                for key, value in parameter_samples[i].items():
                    new_sample[key] = value

            self._samples.append(new_sample)
        
        return self._samples


class CrossProductSampler(BaseSampler):
    """
    Class defining cross-product sampling.

    Its sampler data takes two blocks, ``constants`` and ``parameters``:

    .. code:: yaml

        sampler:
            type: cross_product
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]

    Entries in the ``constants`` dictionary will be added to all samples.
    Entries in the ``parameters`` block will have the cross product taken.
    The above entry sould result in the samples:

    .. code:: python

        [
            {X1: 20, X2: 5, X3: 5},
            {X1: 20, X2: 5, X3: 10},
            {X1: 20, X2: 10, X3: 5},
            {X1: 20, X2: 10, X3: 10}
        ]
    """

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        :returns: True if the schema is valid, False otherwise.
        """
        if not super(CrossProductSampler, self).is_valid():
            return False

        if 'constants' not in self.data and 'parameters' not in self.data:
            LOG.error(
                "Either constants or parameters must be included in the "
                "sampler data"
                )
            return False

        return True

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        parameters = []
        with suppress(KeyError):
            parameters.extend(list(self.data['constants'].keys()))
        with suppress(KeyError):
            parameters.extend(list(self.data['parameters'].keys()))

        return parameters

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

        product_list = []

        with suppress(KeyError):
            product_list.extend(
                [[value] for key, value in self.data['constants'].items()]
            )

        with suppress(KeyError):
            product_list.extend(
                [value for key, value in self.data['parameters'].items()]
            )

        sample_list = itertools.product(*product_list)

        self._samples = []

        for sample in sample_list:
            new_sample = {}
            for i, key in enumerate(self.parameters):
                new_sample[key] = sample[i]
            self._samples.append(new_sample)

        return self._samples


class CsvSampler(BaseSampler):
    """
    Class which reads samples from a csv file.

    Its sampler data takes two blocks, ``csv_file`` and ``row_headers``:

    .. code:: yaml

        sampler:
            type: csv
            csv_file: file_name.csv
            row_headers: True

    The ``csv_file`` entry gives the path to the csv file to read.
    the ``row_headers`` indicates whether the data is one entry per
    row (True) or one entry per column (False).
    """

    def __init__(self, data):
        super(CsvSampler,self).__init__(data)
        self.path = Path(self.data['csv_file'])
        self._csv_data = None

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        :returns: True if the schema is valid, False otherwise.
        """
        if not super(CsvSampler, self).is_valid():
            return False
        if not self.path.is_file():
            LOG.error(f"Could not find file {self.path} for CsvSampler")
            return False

        test_length = None

        for key, value in self.csv_data.items():
            if test_length is None:
                test_length = len(value)
            if len(value) != test_length:
                LOG.error(
                    "All parameters must have the same nuumber of entries"
                    )
                return False

        return True

    @property
    def csv_data(self):
        """
        The csv data as a dictionary of lists.
        """
        if self._csv_data is None:
            csv_data = read_csv(self.path)
            if not self.data['row_headers']:
                csv_data = transpose_tabular(csv_data)
            self._csv_data = {}
            for line in csv_data:
                self._csv_data[line[0]] = line[1:]
        return self._csv_data

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        return list(self.csv_data.keys())

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

        num_samples = 1

        for key, value in self.csv_data.items():
            num_samples = len(value)
            break

        for i in range(num_samples):
            new_sample = {}

            for key, value in self.csv_data.items():
                new_sample[key] = value[i]

            self._samples.append(new_sample)

        return self._samples


class CustomSampler(BaseSampler):
    """
    Class which reads samples from a user-defined python function.

    Its sampler data takes three blocks, ``function``, ``module``, and
    ``args``.

    .. code:: yaml

        sampler:
            type: custom
            function: <name of function>
            module: <path to module containing function>
            args: {} # Dictionary of keyword arguments to pass
                     # To the function.

    The ``function`` entry names the function to call to get the samples.
    It must return a list of dictionaries:

        .. code:: python

            [{'b': 0.89856, 'a': 1}, {'b': 0.923223, 'a': 1}, ... ]

    The ``module`` entry gives the path to the module to call.

    The ``args`` entry contains keyword arguments for the function.
    The sampler will pass these to the function:

    .. code:: python

        samples = custom_function(**args)

    The returned values from the function will be returned from
    get_samples().
    """

    def __init__(self, data):
        super(CustomSampler,self).__init__(data)
        self.path = Path(self.data['module'])
        self._sample_function = None

    def is_valid(self):
        """
        Check if the sampler is valid.

        Checks the sampler data against the built-in schema.

        :returns: True if the schema is valid, False otherwise.
        """
        if not super(CustomSampler, self).is_valid():
            return False
        if not self.path.exists():
            LOG.error(f"Unable to find module {self.path}")
            return False
        if self.sample_function is None:
            return False
        return True

    @property
    def sample_function(self):
        """
        Returns the custom sampling function, importing it if necessary.
        """
        if self._sample_function is None:
            sys.path.append(str(self.path.parent))
            module_name = self.path.name
            if module_name.endswith(".py"):
                module_name = module_name.rsplit(".", 1)[0]

            custom_module = importlib.import_module(module_name)

            try:
                self._sample_function = getattr(
                                            custom_module,
                                            self.data['function']
                                            )
            except AttributeError:
                LOG.error(f"Requested function {self.data['function']}"
                          f" not found in module {self.path} ")

        return self._sample_function

    @property
    def parameters(self):
        """
        Return a of list of the parameters being generated by the
        sampler.
        """
        if self._samples is None:
            _toks = self.get_samples()
        return list(self._samples[0].keys())

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

        self._samples = self.sample_function(**self.data['args'])

        return self._samples


SAMPLE_FUNCTIONS_DICT = {
    # 'best_candidate': BestCandidateSampler,
    'column_list': ColumnListSampler,
    'list': ListSampler,
    'cross_product': CrossProductSampler,
    'csv': CsvSampler,
    'custom': CustomSampler
}

SAMPLE_FUNCTIONS_KEYS = SAMPLE_FUNCTIONS_DICT.keys()
"""list: List of available sampling methods."""

