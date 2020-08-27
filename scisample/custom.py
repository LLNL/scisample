"""
Module defining the custom sampler object.
"""

import importlib
import logging
import sys

from pathlib import Path

from scisample.base_sampler import (BaseSampler)
from scisample.utils import log_and_raise_exception

LOG = logging.getLogger(__name__)


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
        super().__init__(data)
        self.path = Path(self.data['module'])
        self._sample_function = None
    #     self.check_validity()

    # def check_validity(self):
        if not self.path.exists():
            log_and_raise_exception(
                f"Unable to find module {self.path} for 'custom' sampler")
        if self.sample_function is None:
            log_and_raise_exception(
                "The 'custom' sampler requires a 'sample_function'")

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
            self.get_samples()
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
