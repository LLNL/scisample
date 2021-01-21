# scisample

`scisample` is a Python 3 package that implements a number of parameter
sampling methods for scientific computing. Specifications for sampling
are written in  the YAML markup language.

# Installation with a python virtual environment

1. `cd` into the top level scisample directory
1. `python3 -m venv venv`
1. `source venv/bin/activate`
1. `pip install -r requirements.txt`
1. `pip install -e .`

# Documentation

 1. `cd docs` into the top level scisample directory
 1. `make <documentation type>`, where <documentation type> includes 'html', 'latexpdf', 'text', etc.


# Testing

 1. `cd` into the top level scisample directory
 1. `pytest tests`
 1. `pytest --cov=scisample tests/`

# Community 

`scisample` is an open source project. Questions, discussion, and
contributions are welcome. Contributions can be anything from new
packages to bugfixes, documentation, or even new core features.

# Contributing

Contributing to `scisample` is relatively easy. Just send us a pull
request. When you send your request, make `develop` the destination
branch on the scisample repository.

Your PR must pass `scisamples`'s unit tests and documentation tests, and
must pass most `flake8` and `pylint` tests. We enforce these guidelines
with our CI process. Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for
more information. 

# Code of Conduct

Please note that `scisample` has a [Code of Conduct](./CODE_OF_CONDUCT.md). By
participating in the `scisample` community, you agree to abide by its rules.

# Authors

Current authors of `scisample` include Brian Daub, Jessica Semler, Cody
Raskin, & Chris Krenn. 

# License

`scisample` is distributed under the the MIT license.

All new contributions must be made under the MIT license.

Please see [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-815909
