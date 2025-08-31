# EigenpairFlow
<img src="https://github.com/t-uda/eigenpairflow/raw/main/eigenpairflow-logo.png" alt="eigenpairflow-logo" width="256" />

[![CI](https://github.com/t-uda/EigenpairFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/t-uda/EigenpairFlow/actions/workflows/ci.yml)

Continuous and stable tracking of eigendecomposition for parameter-dependent matrices.

See the demonstration notebook here!

* [`toy_model_demonstration.ipynb`](notebooks/toy_model_demonstration.ipynb)
* [`magnitude_demonstration.ipynb`](notebooks/magnitude_demonstration.ipynb)

## Supported Versions

This project is continuously tested against the following versions to ensure compatibility.

The project's dependencies are managed by Poetry, which resolves the most suitable NumPy v1.x version for each Python interpreter. Forward-compatibility with NumPy v2.x is also explicitly tested on newer Python versions.

| Python Version | Tested NumPy Series |
| :------------: | :-----------------: |
| 3.9            | 1.x                 |
| 3.10           | 1.x                 |
| 3.11           | 1.x & 2.x           |
| 3.12           | 1.x & 2.x           |
