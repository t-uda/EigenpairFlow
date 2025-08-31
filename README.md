# EigenpairFlow
<img src="https://github.com/t-uda/eigenpairflow/raw/main/eigenpairflow-logo.png" alt="eigenpairflow-logo" width="256" />

[![CI](https://github.com/t-uda/EigenpairFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/t-uda/EigenpairFlow/actions/workflows/ci.yml)

Continuous and stable tracking of eigendecomposition for parameter-dependent matrices.

## Installation

Install from PyPI:

```bash
pip install eigenpairflow
```

## Usage

Here is a minimal example of tracking the eigenpairs of a parameter-dependent matrix `A(t)`.

First, define the matrix-valued function `A(t)` and its derivative `dA(t)/dt`.

```python
import numpy as np
from eigenpairflow import eigenpairtrack

# Define a matrix function A(t) = R(t) D(t) R(t).T
def A_func(t: float) -> np.ndarray:
    theta = np.pi / 4 * t
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([1, 1 + t])
    return R @ D @ R.T

# Define its derivative dA(t)/dt
def dA_func(t: float) -> np.ndarray:
    theta = np.pi / 4 * t
    dtheta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([1, 1 + t])
    dR = dtheta * np.array([[-s, -c], [c, -s]])
    dD = np.diag([0, 1])
    return dR @ D @ R.T + R @ dD @ R.T + R @ D @ dR.T
```

Then, call `eigenpairtrack` to solve the tracking problem over a given time interval.

```python
# Set the time interval and evaluation points
t_start, t_end = 0.0, 4.0
t_eval = np.linspace(t_start, t_end, 200)

# Track the eigenpairs
results = eigenpairtrack(A_func, dA_func, (t_start, t_end), t_eval)

# The results object contains the tracked eigenvalues (results.ls)
# and eigenvectors (results.Qs) at each time point in t_eval.
print("Tracking successful:", results.success)
print("Eigenvalues at t=0:", results.ls[0])
print("Eigenvectors at t=0:\\n", results.Qs[0])
```

For more detailed examples, please see the [demonstration notebooks](notebooks/).

## Supported Versions

This project is continuously tested against the following versions to ensure compatibility.

The project's dependencies are managed by Poetry, which resolves the most suitable NumPy v1.x version for each Python interpreter. Forward-compatibility with NumPy v2.x is also explicitly tested on newer Python versions.

| Python Version | Tested NumPy Series |
| :------------: | :-----------------: |
| 3.9            | 1.x                 |
| 3.10           | 1.x                 |
| 3.11           | 1.x & 2.x           |
| 3.12           | 1.x & 2.x           |
