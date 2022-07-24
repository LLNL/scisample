""" setup.py """

import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="scisample",
    version="1.0.3",
    author="Brian Daub, Chris Krenn, Cody Raskin, Jessica Semler",
    author_email="daub1@llnl.gov, krenn1@llnl.gov, raskin1@llnl.gov, semler1@llnl.gov",
    description="Parameter sampling for scientific computing",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/LLNL/scisample",
    license='MIT License',
    packages=setuptools.find_packages(),
    package_data = {'scisample': ['config.yaml']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'parse',
        'numpy',
        'cached_property',
    ],
    extras_require={
        'maestrowf': ['maestrowf'],
        'best_candidate': ['pandas', 'scipy']
    },
    scripts=['bin/pgen_scisample.py']
)
