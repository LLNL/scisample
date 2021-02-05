""" setup.py """

import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="scisample",
    version="0.0.1",
    author="Brian Daub, Jessica Semler, Cody Raskin, & Chris Krenn",
    author_email="crkrenn@gmail.com",
    description="Parameter sampling for scientific computing",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/LLNL/scisample",
    license='MIT License',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],
    extras_require={
        'maestrowf': ['maestrowf'],
        'best_candidate': ['pandas', 'numpy', 'scipy']
    },
    scripts=['bin/pgen_scisample.py']
)
