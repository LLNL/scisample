Note
====

This example was copied from the `codepy repo <https://wci-git.llnl.gov/simulations/codepy-projects/codepy/-/tree/main/examples/workflow/sampling>_.

Sampling Input Parameters using the ``scisample`` Package
=========================================================

In order to support the need for generating samples for ensembles and studies,
we have been developing the ``scisample`` package to support different sampling
strategies.  This packages is currently integrated into ``codepy``, but will
eventually become a standalone package that can be used outside of ``codepy``
as well.  This example will demonstrate some of the ways that ``scisample``
can be used with ``codepy`` to generate sample points for ensembles.

.. seealso::
    :py:mod:`codepy.scisample`
        Documentation of ``scisample`` subpackage.

.. note::
    If you are using a local installation of ``codepy`` to run these examples,
    you must also install the
    `codepy-ares <https://wci-git.llnl.gov/simulations/codepy-projects/codepy-ares>`_
    plugin.

The example directory contains several files:

.. code:: text
    
    sample_output/

The files for this example can be found in the
`examples/workflow/sampling <https://wci-git.llnl.gov/simulations/codepy-projects/codepy/-/tree/develop/examples/workflow/sampling`_
directory of the ``codepy`` repository.

The driver file is a simple ``ares`` input deck, which defines default values
for three parameters, which will then be overwritten via different sampling
methods.

List Sampler
************

The list sampler is a basic list of parameter keys and values.  Both constant
values and varying values can be included.  The ``list_sampler_config.yaml``
file contains the basic definition of a list sampler.

.. literalinclude::  ../../../examples/workflow/sampling/list_sampler_config.yaml
    :caption: ``list_sampler_config.yaml`` file defining the list sampler.
    :language: yaml

We take advantage of ``codepy``'s ability to use multiple configuration files
to run with both the ``list_sampler_config.yaml`` and ``codepy_config.yaml``
files.

.. code-block:: text
    :caption: Command to run a study using the list sampler.  The ``-o`` flag
        redirects the output to a local ``studies`` directory.
    
    codepy run model -o studies -c list_sampler_config.yaml codepy_config.yaml

The ``codepy_study.yaml`` file created by this command is shown in
``sample_output/list_sampler_study.yaml``.  The highlighted lines show where the
sample points from the sampler are written to the ``global.parameters``  block.

.. literalinclude:: ../../../examples/workflow/sampling/sample_output/list_sampler_study.yaml
    :language: yaml
    :caption: ``codepy_study.yaml`` file showing the inclusion of parameters from a list sampler.
    :emphasize-lines: 45,47-49,51-51,55-57

If we run ``table`` on the resulting study, we can see the parameters passed down as expected.

.. code-block:: text
    :caption: Output of ``codepy table`` for the list sampler study.  Note that we named the
        study ``list-sampler`` in the ``list_sampler_config.yaml`` file, so by using the
        directory filter flag ``-f``, we can obtain only those directories which have
        ``list-sampler`` in the name.  This will allow us to pull in only the studies of interest
        when running multiple studies in the same directory.  The logging output has been excluded
        from the text below.

    codepy table studies -f list-sampler

                      __
      _________  ____/ /__  ____  __  __
     / ___/ __ \/ __  / _ \/ __ \/ / / /
    / /__/ /_/ / /_/ /  __/ /_/ / /_/ /
    \___/\____/\__,_/\___/ .___/\__, /
                        /_/    /____/

    Kernel                    X1    X2    X3
    ----------------------  ----  ----  ----
    X1.20.X2.10.X3.10_xxxx    20    10    10
    X1.20.X2.5.X3.5_xxxx      20     5     5

As we can see, two sample points were created, with ``X1`` set to a constant,
and ``X2`` and ``X3`` set based on the config file input.

Cross Product Sampler
*********************

The list sampler is a basic list of parameter keys and values.  Both constant
values and varying values can be included.  The ``cross_product_config.yaml``
file contains the basic definition of a cross product sampler.

.. literalinclude::  ../../../examples/workflow/sampling/cross_product_config.yaml
    :caption: ``cross_product_config.yaml`` file defining the cross product sampler.
    :language: yaml

We take advantage of ``codepy``'s ability to use multiple configuration files
to run with both the ``cross_product_config.yaml`` and ``codepy_config.yaml``
files.

.. code-block:: text
    :caption: Command to run a study using the cross product sampler.  The ``-o`` flag
        redirects the output to a local ``studies`` directory.
    
    codepy run model -o studies -c cross_product_config.yaml codepy_config.yaml

The ``codepy_study.yaml`` file created by this command is shown in
``sample_output/cross_product_study.yaml``.  The highlighted lines show where the
sample points from the sampler are written to the ``global.parameters``  block.

.. literalinclude:: ../../../examples/workflow/sampling/sample_output/cross_product_study.yaml
    :language: yaml
    :caption: ``codepy_study.yaml`` file showing the inclusion of parameters from a cross product sampler.
    :emphasize-lines: 45,47-49,51-51,55-57

If we run ``table`` on the resulting study, we can see the parameters passed down as expected.

.. code-block:: text
    :caption: Output of ``codepy table`` for the list sampler study.  Note that we named the
        study ``cross-product`` in the ``cross_product_config.yaml`` file, so by using the
        directory filter flag ``-f``, we can obtain only those directories which have
        ``cross-product`` in the name.  This will allow us to pull in only the studies of interest
        when running multiple studies in the same directory.  The logging output has been excluded
        from the text below.

    codepy table studies -f cross-product

                      __
      _________  ____/ /__  ____  __  __
     / ___/ __ \/ __  / _ \/ __ \/ / / /
    / /__/ /_/ / /_/ /  __/ /_/ / /_/ /
    \___/\____/\__,_/\___/ .___/\__, /
                        /_/    /____/
    Kernel                    X1    X2    X3
    ----------------------  ----  ----  ----
    X1.20.X2.10.X3.10_xxxx    20    10    10
    X1.20.X2.10.X3.5_xxxx     20    10     5
    X1.20.X2.5.X3.10_xxxx     20     5    10
    X1.20.X2.5.X3.5_xxxx      20     5     5

As we can see, four sample points were created, with ``X1`` set to a constant,
and ``X2`` and ``X3`` set as a cross product of their parameter values.

Column List Sampler
*******************

The column list sampler is similar to the list sampler, but takes as its input a
text block containing samples in column format.  As with the list sampler, constant
values can also be included.  The ``column_list_config.yaml``
file contains the basic definition of a column list sampler.

.. literalinclude::  ../../../examples/workflow/sampling/column_list_config.yaml
    :caption: ``column_list_config.yaml`` file defining the column_list sampler.
    :language: yaml

We take advantage of ``codepy``'s ability to use multiple configuration files
to run with both the ``column_list_config.yaml`` and ``codepy_config.yaml``
files.

.. code-block:: text
    :caption: Command to run a study using the column_list sampler.  The ``-o`` flag
        redirects the output to a local ``studies`` directory.
    
    codepy run model -o studies -c column_list_config.yaml codepy_config.yaml

The ``codepy_study.yaml`` file created by this command is shown in
``sample_output/column_list_study.yaml``.  The highlighted lines show where the
sample points from the sampler are written to the ``global.parameters``  block.

.. literalinclude:: ../../../examples/workflow/sampling/sample_output/column_list_study.yaml
    :language: yaml
    :caption: ``codepy_study.yaml`` file showing the inclusion of parameters from a column list sampler.
    :emphasize-lines: 45,47-49,51-51,55-57

If we run ``table`` on the resulting study, we can see the parameters passed down as expected.

.. code-block:: text
    :caption: Output of ``codepy table`` for the column list study.  Note that we named the
        study ``column-list`` in the ``column_list_config.yaml`` file, so by using the
        directory filter flag ``-f``, we can obtain only those directories which have
        ``column-list`` in the name.  This will allow us to pull in only the studies of interest
        when running multiple studies in the same directory.  The logging output has been excluded
        from the text below.

    codepy table studies -f column-list

                      __
      _________  ____/ /__  ____  __  __
     / ___/ __ \/ __  / _ \/ __ \/ / / /
    / /__/ /_/ / /_/ /  __/ /_/ / /_/ /
    \___/\____/\__,_/\___/ .___/\__, /
                        /_/    /____/

    Kernel                    X1    X2    X3
    ----------------------  ----  ----  ----
    X1.20.X2.10.X3.10_xxxx    20    10    10
    X1.20.X2.5.X3.5_xxxx      20     5     5

As we can see, two sample points were created, with ``X1`` set to a constant,
and ``X2`` and ``X3`` set based on the config file input.

Csv Sampler
***********

The csv sampler is similar to the list sampler, but takes as its input a
path to a comma separated values file.  Both row headers and column headers
are supported, and the example files ``row_headers.csv`` and
``column_headers.csv`` will produce the same output, and the ``csv_config.yaml``
file has both configuration in it.

.. literalinclude::  ../../../examples/workflow/sampling/csv_config.yaml
    :caption: ``csv_config.yaml`` file defining the csv sampler.
    :language: yaml

We take advantage of ``codepy``'s ability to use multiple configuration files
to run with both the ``csv_config.yaml`` and ``codepy_config.yaml``
files.

.. code-block:: text
    :caption: Command to run a study using the csv sampler.  The ``-o`` flag
        redirects the output to a local ``studies`` directory.
    
    codepy run model -o studies -c csv_config.yaml codepy_config.yaml

The ``codepy_study.yaml`` file created by this command is shown in
``sample_output/csv_study.yaml``.  The highlighted lines show where the
sample points from the sampler are written to the ``global.parameters``  block.

.. literalinclude:: ../../../examples/workflow/sampling/sample_output/csv_study.yaml
    :language: yaml
    :caption: ``codepy_study.yaml`` file showing the inclusion of parameters from a csv sampler.
    :emphasize-lines: 45,47-49,51-51,55-57

If we run ``table`` on the resulting study, we can see the parameters passed down as expected.

.. code-block:: text
    :caption: Output of ``codepy table`` for the csv study.  Note that we named the
        study ``csv-sampler`` in the ``csv_config.yaml`` file, so by using the
        directory filter flag ``-f``, we can obtain only those directories which have
        ``csv-sampler`` in the name.  This will allow us to pull in only the studies of interest
        when running multiple studies in the same directory.  The logging output has been excluded
        from the text below.

    codepy table studies -f csv-sampler

                      __
      _________  ____/ /__  ____  __  __
     / ___/ __ \/ __  / _ \/ __ \/ / / /
    / /__/ /_/ / /_/ /  __/ /_/ / /_/ /
    \___/\____/\__,_/\___/ .___/\__, /
                        /_/    /____/

    Kernel                    X1    X2    X3
    ----------------------  ----  ----  ----
    X1.20.X2.10.X3.10_xxxx    20    10    10
    X1.20.X2.5.X3.5_xxxx      20     5     5

As we can see, two sample points were created, with ``X1`` set to a constant,
and ``X2`` and ``X3`` set based on the config file input.

Full Content of Example Files
*****************************

.. literalinclude::  ../../../examples/workflow/sampling/model/phase1_driver
    :caption: Simple ``ares`` deck that defines default values.

.. literalinclude::  ../../../examples/workflow/sampling/model/model.yaml
    :caption: ``model.yaml`` file which configures the example model.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/list_sampler_config.yaml
    :caption: ``list_sampler_config.yaml``, a codepy configuration file for using the list sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/sample_output/list_sampler_study.yaml
    :caption: Study file generated using the list sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/cross_product_config.yaml
    :caption: ``cross_product_config.yaml``, a codepy configuration file for using the list sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/sample_output/cross_product_study.yaml
    :caption: Study file generated using the cross product sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/column_list_config.yaml
    :caption: ``column_list_config.yaml``, a codepy configuration file for using the column list sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/sample_output/column_list_study.yaml
    :caption: Study file generated using the column list sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/csv_config.yaml
    :caption: ``csv_config.yaml``, a codepy configuration file for using the csv sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/sample_output/csv_study.yaml
    :caption: Study file generated using the csv sampler.
    :language: yaml

.. literalinclude::  ../../../examples/workflow/sampling/row_headers.csv
    :caption: ``row_headers.csv``, a csv file containing samples.

.. literalinclude::  ../../../examples/workflow/sampling/column_headers.csv
    :caption: ``column_headers.csv``, a csv file containing samples.

.. literalinclude::  ../../../examples/workflow/sampling/codepy_config.yaml
    :caption: A ``codepy_config.yaml`` file for running interactively.  These
        settings disable batch submission and MPI, and should only be used for
        simple examples.  It additionally includes settings for ``codepy table``
        which will be used repeatedly in this example.

