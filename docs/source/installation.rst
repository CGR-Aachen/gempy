.. role:: raw-html-m2r(raw)
   :format: html


Installation
------------


Installing ``GemPy``
^^^^^^^^^^^^^^^^^^^^^^^^

We provide the latest release version of ``GemPy`` via **PyPi** package services. We highly
recommend using `PyPi <https://pypi.org/project/gempy/>`_,

``$ pip install gempy``

as it will take care of automatically installing all the required dependencies - except in
windows that required one extra step.

Windows Installation
~~~~~~~~~~~~~~~~~~~~

Windows does not have a **gcc compilers** pre-installed. The easiest way to get a ``theano``
compatible compiler is by using the ``theano`` conda installation. Therefore the process
would be the following:

``$ conda install theano``

``$ pip install gempy``

**Notes:**

- The conda version of ``theano`` comes with a non critical bug that will rise a warning (``scan_perform.c``)
  when computing a model with gempy. Once the compiler is installed, installing the pip version of ``theano``
  will solve the problem:

``$ pip install theano --force-reinstall``

- The conda version of ``theano`` is not compatible with Catalina MacOS. Use pip!

- For a more detailed description on the installation in case
  something does not work or for CUDA acceleration check `Windows installation guide (March 2020)`_ and
  `MacOS installation guide (May 2020)`_

Developers Installation
~~~~~~~~~~~~~~~~~~~~~~~

If you are planning to contribute in ``gempy`` the easiest way is to clone the
repository from `GitHub <https://github.com/cgre-aachen/gempy>`_ and use

``$ pip install -e .``

on the repository root. Notice that on the repo you can also find a ``optional-requirements.txt``
for more experimental functionality. Finally to compile ``sphinx`` you will need:

``$ pip install sphinx, sphinx-gallery``


Manual installation
^^^^^^^^^^^^^^^^^^^
If you want to use the newest, cutting-edge version of ``GemPy`` you can clone the current repository by downloading it manually or by using Git by calling

``$ git clone https://github.com/cgre-aachen/gempy.git``

and then manually install it using the provided Python install file by calling

``$ python gempy/setup.py install``

in the cloned or downloaded repository folder.

Alternatively to running ``setup.py``, you can use pip to handle the installation from the repository and the updating of the path variables.
For this, navigate to the root of the cloned repository and run

``$ pip install -e .``

Make sure you have installed all necessary dependencies listed below before using ``GemPy``.


Dependencies
^^^^^^^^^^^^

``GemPy`` requires Python 3.x and makes use of numerous open-source libraries:

* pandas
* Theano>=1.0.4
* matplotlib
* numpy
* pytest
* seaborn>=0.9
* networkx
* scikit-image>=0.17
* pyvista

Optional requirements:

* gdal
* qgrid==1.3.0
* pymc3
* pyevtk
* pyqrcode
* mplstereonet

Overall we recommend the use of a **dedicated Python distribution**\ , such as
`Anaconda <https://www.continuum.io/what-is-anaconda>`_\ , for hassle-free package installation. 
We are currently working on providing ``GemPy`` also via Anaconda Cloud, for easier installation of
its dependencies.

Conflictive packages.
~~~~~~~~~~~~~~~~~~~~~

Installing Theano (especially under Windows) and vtk can sometimes be difficult.
Here, we provide advice that should use in most cases (but certainly not all):


* ``Theano``\ :
    install the following packages before installing theano:

    ``$ conda install mingw libpython m2w64-toolchain``

    Then install Theano via

    ``$ conda install theano``

    If the installation fails at some point try to re-install anaconda for a single user (no administrator privileges)
    and with the Path Environment set.
    To use Theano with ``numpy version 1.16.0`` or following, it has to be updated to ``Theano 1.0.4`` using

    ``$ pip install theano --upgrade``

    Note that this is not yet available in the conda package manager.


* ``vtk`` :
    There have been compatibility problems between with the ``vtk`` package
    and python 3.8. The simplest solution to install it is to
    use ``$ conda install python==3.6`` to downgrade the python version and then
    using ``$ pip install vtk`` .



Docker
^^^^^^

Finally we also provide precompiled Docker images hosted on Docker Hub with all necessary dependencies to get
GemPy up and running (\ **except vtk**\ ).

Docker is an operating-system-level-visualization software,
meaning that we can package a tiny operating system with pre-installed
software into a Docker image. This Docker image can then be shared
with and run by others, enabling them to use intricate dependencies
with just a few commands. For this to work the user needs to have a
working `Docker <https://www.docker.com/>`_ installation.


Pull Docker image from DockerHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to get `gempy` running is by running the pre-compiled Docker image (containing everything you
need) directly from the cloud service Docker Hub to get a locally running Docker container. Make sure to set your 
Docker daemon to Linux containers in Docker's context menu.

``$ docker run -it -p 8899:8899 leguark/gempy``


This will automatically pull the Docker image from Docker Hub and run it, opening a command line shell inside of the
running Docker container. There you have access to the file system inside of the container. Note that this pre-compiled
Docker image already contains the GemPy repository. 

Once you are in the docker console if you want to open the tutorials you will need to run:

``$ jupyter notebook --ip 0.0.0.0 --port 8899 --no-browser --allow-root``


Notice that we are running the notebook on the port  8899 to try to avoid conflicts with jupyter servers running in
your system. If everything worked fine, the address to the jupyter notebook will be display on the console. It
has to look something like this (Just be aware of the  brackets):

.. code-block::

   To access the notebook, open this file in a browser:
           file:///root/.local/share/jupyter/runtime/nbserver-286-open.html
   Or copy and paste one of these URLs:
       http://(ce2cdcc55bb0 or 127.0.0.1):8899/?token=97d52c1dc321c42083d8c1b4d




Windows installation guide (March 2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. This step is **only important if you want GPU acceleration**. Install CUDA if you do not have it already.

   * For CUDA > 10 (For RTX cards you need those drivers):

       - Go to your cuda installation (probably ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin`` )

       - Duplicate ``cublas64_XX`` and ``nvrtc64_XX`` and rename them to ``cublas64_70`` and ``nvrtc64_70``\ .


#. Install Conda (recommended: latest miniconda)

    #. Use conda prompt as the python terminal

        Install Anaconda with the options  "for current user".

    Add conda enviroment:

    ``$ conda create --name gempy python==3.7``

     or

    #. Set up conda in the powershell
        Install Anaconda with the options  "for current user" and "add conda to Path environment".

        ``$ conda init powershell``

        **As admin:** ``$ Set-ExecutionPolicy RemoteSigned``

        After this stage we should have a new empty environment attached to a user


#. Install Theano and associated packages from the Anaconda prompt as administrator:

    ``$ conda update --all``

    ``$ conda install libpython``

    ``$ conda install m2w64-toolchain``

    ``$ conda install git``

    ``$ conda install -c conda-forge pygpu``


    ``$ pip install theano==1.0.4``



#. Install ``GemPy``

    install the latest release version of ``GemPy`` via ``PyPi``:

    ``$ pip install gempy``

    Alternatively, if you need the latest developments in GemPy, follow the instruction from the chapter **Manual Installation** instead.

#. Set up Jupyter to work properly with conda environments:

    ``$ conda install Jupyter``

    ``$ conda install nb_conda_kernels``

    ``$ pip install jupyter-conda``


#. Optional requirements:


    ``$ pip install pyevtk``

    ``$ conda install gdal``


**Note**\ :


* some other packages required by Theano are already included in Anaconda: numpy, scipy, mkl-service, nose, and sphinx.
* ``pydot-ng`` (suggested on Theano web site) yields a lot of errors, therefore we dropped this.  It is needed to
handle large picture for gif/images and probably it is not needed by GemPy.
* Trying to install all the packages in one go does not work, as well as doing the same in  Anaconda Navigator, or
installing an older Anaconda release with Python 3.5 (Anaconda3 4.2.0) as indicated in some tutorial on Theano.


MacOS installation guide (May 2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Note**\ : The following guide is for a standard installation (no GPU support).
It should work on MacOS 10.14 as well as 10.15 (Catalina).

#. Install Anaconda (**Python Version 3.7**\ )

    For a minimal installation, you can install the
    `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html|>`_\.
    The full Anaconda distribution contains some additional features, IDE's
    etc. and is available on the `Anaconda page <https://www.anaconda.com/products/individual|>`_\.

#. Create a ``GemPy`` conda environment

    We strongly suggest to create a separate conda environment, to avoid
    conflicts with other Python installations and packages on your system.
    This is easily done in a bash terminal:

    ``$ conda create --name gempy python==3.7``

    Set up Jupyter to work properly with conda environments:

    ``$ python -m ipykernel install --user --name gempy``

    Activate the new environment (do this every time you create a new terminal session):

    ``$ conda activate gempy``

    You should now see `(gempy)` at the beginning of the command line. If
    the previous command fails (some known issues), then try:

    ``$ source activate gempy``


#. Install the Xcode command-line tools

    In order for ``Theano`` to access the system compilers on MacOS, the Xcode command-line tools are required.
    To automatically install the correct version for your OS, run:

    ``$ xcode-select --install``

    Follow the installation instructions. After the installation is complete, open ``Software  Update`` from your ``System Preferences`` and install any available  updates for the command-line tools.



#. Install required Python packages

    ``$ conda update --all``

    Install theano via PyPi

    ``$ pip install theano==1.0.4``

    Test the `theano` installation: run ``python``\ , then try ``import theano``\ .
    If you get an error (e.g. ``stdio.h`` not found), then:

    Test if the Xcode command-line tools are correctly installed and up-to-date.(info for
    example `here <https://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x>`_).
    If this still fails, try installing ``theano`` through conda-forge instead:

    ``$ conda install -c conda-forge theano``


#. ``Install GemPy``

    install the latest release version of ``GemPy`` via ``PyPi``:

    ``$ pip install gempy``

    Alternatively, if you need the latest developments in GemPy, follow the instruction from the chapter **Manual Installation** instead.



#. Install optional requirements:

    ``$ pip install pyvista``

    ``$ pip install pyevtk``

    ``$ conda install gdal``



