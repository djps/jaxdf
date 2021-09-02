# jaxdiff

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![codecov](https://codecov.io/gh/astanziola/jwave/branch/main/graph/badge.svg?token=6J03OMVJS1)](https://codecov.io/gh/astanziola/jwave)
![Continous Integration](https://github.com/astanziola/jwave/actions/workflows/ci-build.yml/badge.svg)

[**Overview**](#overview)
| [**Installation**](#installation)

## Overview

jaxdiff is a [JAX](https://jax.readthedocs.io/en/stable/)-based package defining a research framework for writing differentiable numerical simulators with arbitrary discretizations. 

The intended use is to build numerical models of physical systems, such as wave propagation, or the numerical solution of partial differential equations, that are easy to customize to the user's research needs. Such models are pure functions that can be included into arbitray differentiable programs written in [JAX](https://jax.readthedocs.io/en/stable/). For example, they can be used as layers of neural networks, or to build a physics loss function.

## Installation

Before installing `jaxdiff`, make sure that [you have installed JAX](https://github.com/google/jax#installation). Follow the instruction to install JAX with NVidia GPU support if you want to use `jwave` on the GPUs. 

Install jwave by `cd` in the repo folder an run
```bash
pip install -r requirements.txt
pip install .
```

If you want to run the notebooks, you should also install the following packages
```bash
pip install jupyterlab, tqdm
```

3. [`deepXDE`](https://deepxde.readthedocs.io/en/latest/): a TensorFlow and PyTorch library for scientific machine learning.
4. [`SciML`](https://sciml.ai/): SciML is a NumFOCUS sponsored open source software organization created to unify the packages for scientific machine learning. 