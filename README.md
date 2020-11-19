# scisample

A package that implements a number of parameter sampling methods for
scientific computing.

# Installation with a python virtual environment
1. `cd` into the top level scisample directory
1. `python3 -m venv venv`
1. `source venv/bin/activate`
1. `pip install -r requirements.txt`
1. `pip install -e .`

# Testing
 1. `cd` into the top level scisample directory
 1. `pytest tests`
 1. `pytest --cov=scisample tests/`

publish to github/llnl:

git remote add origin https://github.com/LLNL/scisample.git
git branch -M main
git push -u origin main

and, look at files in llnl_github (.github) repo
