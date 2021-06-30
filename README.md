# <p align="center"><img src="docs/readme_images/header_combined_slim.png" width="1000"></p>

> Open-source, implicit 3D structural geological modeling in Python.

[![PyPI](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-1.0-blue.svg)](https://pypi.org/project/gempy/)
[![license: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-blue.svg)](https://github.com/cgre-aachen/gempy/blob/master/LICENSE)
[![Documentation Status](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](http://docs.gempy.org)
[![Travis Build](https://travis-ci.org/cgre-aachen/gempy.svg?branch=master)](https://travis-ci.org/github/cgre-aachen/gempy/branches)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cgre-aachen/gempy/master)
[![DOI](https://zenodo.org/badge/96211155.svg)](https://zenodo.org/badge/latestdoi/96211155)
[![DOCKER](https://img.shields.io/docker/cloud/automated/leguark/gempy.svg)](https://cloud.docker.com/repository/docker/leguark/gempy)

## Overview

[GemPy](https://www.gempy.org/) is a Python-based, **open-source geomodeling library**. It is
capable of constructing complex **3D geological models** of folded
structures, fault networks and unconformities, based on the underlying
powerful **implicit representation** approach. 

## Installation

We provide the latest release version of GemPy via PyPi package services. We highly recommend using PyPi,

`$ pip install gempy`

as it will take care of automatically installing all the required dependencies - except in windows that requires one extra step.

Windows does not have a gcc compilers pre-installed. The easiest way to get a theano compatible compiler is by using 
the theano conda installation. Therefore the process would be the following:

`$ conda install theano`

`$ pip install gempy`

For more information, refer to the [installation documentation](https://docs.gempy.org/installation.html).

## Resources

After installation you can either check the [notebook tutorials](https://docs.gempy.org/getting_started/get_started.html#sphx-glr-getting-started-get-started-py) 
or the [video introduction](https://www.youtube.com/watch?v=n0btC5Zilyc) to get started.

Go to the [documentation site](http://docs.gempy.org/) for further information and enjoy the [tutorials and examples](https://www.gempy.org/tutorials).

For questions and support, please use [discussions](https://github.com/cgre-aachen/gempy/discussions).

If you find a bug or have a feature request, create an [issue](https://github.com/cgre-aachen/gempy/issues).

Follow these [guidelines](https://github.com/cgre-aachen/gempy/blob/WIP_readme-update-march21/CONTRIBUTING.md) to contribute to GemPy.

<a name="ref"></a>
## References 

* de la Varga, M., Schaaf, A., and Wellmann, F. (2019). [GemPy 1.0: open-source stochastic geological modeling and inversion](https://gmd.copernicus.org/articles/12/1/2019/gmd-12-1-2019.pdf), Geosci. Model Dev., 12, 1-32.
* Wellmann, F., & Caumon, G. (2018). [3-D Structural geological models: Concepts, methods, and uncertainties.](https://hal.univ-lorraine.fr/hal-01921494/file/structural_models_for_geophysicsHAL.pdf) In Advances in Geophysics (Vol. 59, pp. 1-121). Elsevier.
* Calcagno, P., Chilès, J. P., Courrioux, G., & Guillen, A. (2008). Geological modelling from field data and geological knowledge: Part I. Modelling method coupling 3D potential-field interpolation and geological rules. Physics of the Earth and Planetary Interiors, 171(1-4), 147-157.
* Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation. Mathematical Geology, 29(4), 571-584.

## Publications using GemPy

* Schaaf, A., de la Varga, M., Wellmann, F., & Bond, C. E. (2021). [Constraining stochastic 3-D structural geological models with topology information using approximate Bayesian computation in GemPy 2.1](https://gmd.copernicus.org/articles/14/3899/2021/gmd-14-3899-2021.html). Geosci. Model Dev., 14(6), 3899-3913. doi:10.5194/gmd-14-3899-2021
* Güdük, N., de la Varga, M. Kaukolinna, J. and Wellmann, F. (in review). Model-Based Probabilistic Inversion Using Magnetic Data: A Case Study on the Kevitsa Deposit.
* Stamm, F. A., de la Varga, M., and Wellmann, F. (2019). [Actors, actions, and uncertainties: optimizing decision-making based on 3-D structural geological models](https://se.copernicus.org/articles/10/2015/2019/se-10-2015-2019.html), Solid Earth, 10, 2015–2043.
* Wellmann, F., Schaaf, A., de la Varga, M., & von Hagke, C. (2019). [From Google Earth to 3D Geology Problem 2: Seeing Below the Surface of the Digital Earth](
https://www.sciencedirect.com/science/article/pii/B9780128140482000156).
In Developments in Structural Geology and Tectonics (Vol. 5, pp. 189-204). Elsevier.

## Gallery

### Geometries

<p>
<table>
<tr>

  <td>
  <a href="https://docs.gempy.org/examples/geometries/1_horizontal_stratigraphic.html#sphx-glr-examples-geometries-1-horizontal-stratigraphic-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model1_nodata.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/geometries/2_fold.html#sphx-glr-examples-geometries-2-fold-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model2_nodata.png" width="300" />
  </a>
  </td>
  
   <td>
  <a href="https://docs.gempy.org/examples/geometries/3_recumbent_fold.html#sphx-glr-examples-geometries-3-recumbent-fold-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model3_nodata.png" width="300" />
  </a>
  </td>

</tr>
<tr>

  <td>
  <a href="https://docs.gempy.org/examples/geometries/4_pinchout.html#sphx-glr-examples-geometries-4-pinchout-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model4_nodata.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/geometries/5_fault.html#sphx-glr-examples-geometries-5-fault-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model5_nodata.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/geometries/6_unconformity.html#sphx-glr-examples-geometries-6-unconformity-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model6_nodata.png" width="300" />
  </a>
  </td>

</tr>
</table>
</p>

### Features

<p>
<table>
<tr>

  <td>
  <a href="https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_3b_cross_sections.html#sphx-glr-tutorials-ch1-fundamentals-ch1-3b-cross-sections-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/sectiontest.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_7_3d_visualization.html#sphx-glr-tutorials-ch1-fundamentals-ch1-7-3d-visualization-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/data_vis.png" width="300" />
  </a>
  </td>
  
   <td>
  <a href="https://docs.gempy.org/examples/geometries/7_combination.html#sphx-glr-examples-geometries-7-combination-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/scalarfield.png" width="300" />
  </a>
  </td>

</tr>
<tr>

  <td>
  <a href="https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_3b_cross_sections.html#sphx-glr-tutorials-ch1-fundamentals-ch1-3b-cross-sections-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/geomap.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/tutorials/ch4-Topology/ch4-1-Topology.html#sphx-glr-tutorials-ch4-topology-ch4-1-topology-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/topology.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/tutorials/ch4-Topology/ch4-1-Topology.html#sphx-glr-tutorials-ch4-topology-ch4-1-topology-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/topology_matrix.png" width="300" />
  </a>
  </td>

</tr>
</table>
</p>


### Case studies

<p>
<table>
<tr>

  <td>
  <a href="https://docs.gempy.org/examples/real/Alesmodel.html#sphx-glr-examples-real-alesmodel-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/alesmodel.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/real/Perth_basin.html#sphx-glr-examples-real-perth-basin-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/perthmodel.png" width="300" />
  </a>
  </td>
  
   <td>
  <a href="https://docs.gempy.org/examples/real/Greenstone.html#sphx-glr-examples-real-greenstone-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/greenstonemodel.png" width="300" />
  </a>
  </td>

</tr>
</table>
</p>
