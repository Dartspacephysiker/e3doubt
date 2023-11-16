Overview
========

EISCAT_3D Uncertainty Estimation (E3Doubt)

E3Doubt is a tool for calculating an order-of-magnitude (hopefully!) estimate of the uncertainties in the plasma parameters measured using the incoherent scatter radar technique with different beam configurations and radar system parameters. It is designed with EISCAT_3D in mind, but is readily adaptable to other radar configurations.   

We recommend using the examples to learn how to use E3Doubt.

Prerequisites
=======

- git >= 2.13
- R >= 3.4
- CMake >= 3(?) (Tested with CMake 3.22.0)


Install
=======

(NB: In the below, if you do not have mamba, replace `mamba` with `conda`)

Option 0: using pip directly
----------------------------

The package may someday be pip-installable from GitHub directly with::

    pip install "e3doubt @ git+https://github.com/Dartspacephysiker/e3doubt.git@main"

This could also be done within a minimal conda environment created with, e.g. ``mamba create -n e3doubt python=3.10 fortran-compiler``

Option 1: mamba/conda install
---------------------------------------------------------------

Get the code, create a suitable conda environment, then use pip to install the package in editable (development) mode::

    git clone https://github.com/Dartspacephysiker/e3doubt
    mamba env create -f e3doubt/binder/environment.yml -n e3doubt
    mamba activate e3doubt
    pip install --editable ./e3doubt

Editable mode (``-e`` or ``--editable``) means that the install is directly linked to the location where you cloned the repository, so you can edit the code.

Note that in this case, the ``deps-from-github`` option means that Ilkka Virtanen's ``ISgeometry`` package will be installed directly from the source on GitHub. (THIS HAS NOT BEEN TESTED)


FIRST RUN
===========
Once you've got everything downloaded, you'll need to run something like the following to make rpy2 aware of the ISgeometry R package. NOTE: You *must* have R installed on your machine for this to work

.. code-block:: python

    # Location of e3d install directory
    e3ddir = '/path/to/e3doubt/'
    
    # Import things we need from rpy2
    from rpy2.robjects.packages import importr
    from rpy2 import robjects as robj
    import os
    
    # base = importr('base')
    utils = importr('utils')
    
    # Install dependencies for ISgeometry package 
    utils.install_packages("maps")
    utils.install_packages("mapdata")

    utils.install_packages(os.path.join(e3ddir,'external/ISgeometry'),
                           repos=robj.NULL,
                           dependencies=True,
                           type="source")


If you get errors about R not being able to locate ISgeometry, make sure that the ``external/ISgeometry/`` directory is not empty:

.. code-block::
   cd /path/to/e3doubt

   # Pull in ISgeometry package
   git submodule update --recursive --remote



Dependencies
============
You should have the following modules installed (this is handled automatically when e3doubt is install using the mamba/conda environment.yml file mentioned above):

- `apexpy <https://github.com/aburrell/apexpy/>`_
- `iri2016 <https://github.com/space-physics/iri2016>`_
- matplotlib
- numpy
- pandas
- `ppigrf <https://github.com/klaundal/ppigrf/>`_ (install with pip install ppigrf)
- `pymsis <https://github.com/swxtrec/pymsis>`_
- rpy2



