# to update scisample on pypi

# checkout master branch
git checkout master

# use a clean venv:
deactivate
source venv_scisample_dev/bin/activate

# bump version:
tbump <new_version>

# remove old wheels:
/bin/rm -rf dist/*

# build wheels:
python -m build

# upload to pypi:
twine upload --verbose dist/*


