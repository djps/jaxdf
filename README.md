# jaxdf - JAX-based Discretization Framework

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) ![Continous Integration](https://github.com/djps/jaxdf/actions/workflows/main.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/djps/jaxdf/badge.svg?branch=main)](https://coveralls.io/github/djps/jaxdf?branch=main)


<!-- [![codecov](https://codecov.io/gh/ucl-bug/jaxdf/branch/main/graph/badge.svg?token=FIUYOCFDYL)](https://codecov.io/gh/ucl-bug/jaxdf) -->

<!-- [![Documentation](https://github.com/djps/jaxdf/actions/workflows/build_docs.yml/badge.svg)](https://djps.github.io/jaxdf/) -->

[Overview](#overview)
| [Example](#example)
| [Installation](#installation)
| [**UCL** Documentation](https://ucl-bug.github.io/jaxdf/)

<br/>

## Overview

jaxdf is a [JAX](https://jax.readthedocs.io/en/stable/)-based package defining a coding framework for writing differentiable numerical simulators with arbitrary discretizations.

The intended use is to build numerical models of physical systems, such as wave propagation, or the numerical solution of partial differential equations, that are easy to customize to the user's research needs. Such models are pure functions that can be included into arbitray differentiable programs written in [JAX](https://jax.readthedocs.io/en/stable/): for example, they can be used as layers of neural networks, or to build a physics loss function.

This is a branch fixed to use `jax==0.2.29` and `jaxlib==0.1.77` which was compiled on Windows 11 with
* Visual Studio 2019 (version 16.11.16)
* Cuda 11.3
* cuDNN 8.2.1
* python 3.7

with a NVIDIA GTX 1650 graphs card.

<br/>

## Example

The following script builds the non-linear operator **(∇<sup>2</sup> + sin)**, using a Fourier spectral discretization on a square 2D domain, and uses it to define a loss function whose gradient is evaluated using JAX Automatic Differentiation.


```python
from jaxdf import operators as jops
from jaxdf import FourierSeries, operator
from jaxdf.geometry import Domain
from jax import numpy as jnp
from jax import jit, grad


# Defining operator
@operator
def custom_op(u):
  grad_u = jops.gradient(u)
  diag_jacobian = jops.diag_jacobian(grad_u)
  laplacian = jops.sum_over_dims(diag_jacobian)
  sin_u = jops.compose(u)(jnp.sin)
  return laplacian + sin_u

# Defining discretizations
domain = Domain((128, 128), (1., 1.))
parameters = jnp.ones((128,128,1))
u = FourierSeries(parameters, domain)

# Define a differentiable loss function
@jit
def loss(u):
  v = custom_op(u)
  return jnp.mean(jnp.abs(v.on_grid)**2)

gradient = grad(loss)(u) # gradient is a FourierSeries
```

<br/>

## Installation

Before installing `jaxdf`, make sure that [you have installed JAX](https://github.com/google/jax#installation). Follow the instruction to install JAX with NVidia GPU support if you want to use `jaxdf` on the GPUs.

Install jaxdf by cloning the repository or downloading and extracting the compressed archive. Then navigate in the root folder in a terminal, and run
```bash
pip install -r .requirements/requirements.txt
pip install .
```

<br/>

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-2111.05218-b31b1b.svg?style=flat)](https://arxiv.org/abs/2111.05218)

This package will be presented at the [Differentiable Programming workshop](https://diffprogramming.mit.edu/) at NeurIPS 2021.

```bibtex
@article{stanziola2021jaxdf,
    author={Stanziola, Antonio and Arridge, Simon and Cox, Ben T. and Treeby, Bradley E.},
    title={A research framework for writing differentiable PDE discretizations in JAX},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}
```

<br/>


- Some of the packaging of this repository is done by editing [this](https://github.com/rochacbruno/python-project-template) templace from @rochacbruno
- The multiple-dispatch method employed is based on `plum`: https://github.com/wesselb/plum

## Related projects

1. [`odl`](https://github.com/odlgroup/odl) Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems.
2. [`deepXDE`](https://deepxde.readthedocs.io/en/latest/): a TensorFlow and PyTorch library for scientific machine learning.
3. [`SciML`](https://sciml.ai/): SciML is a NumFOCUS sponsored open source software organization created to unify the packages for scientific machine learning.
