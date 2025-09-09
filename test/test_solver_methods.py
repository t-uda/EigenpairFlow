import numpy as np
import pytest
from eigenpairflow import eigenpairtrack


def _basic_matrix(t: float) -> np.ndarray:
    return np.array([[2.0, t], [t, 3.0]])


def _d_basic_matrix(_t: float) -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]])


@pytest.mark.parametrize(
    "solver_method,dense_output",
    [
        (None, False),
        ("Euler", False),
        ("RK23", False),
        ("RK23", True),
        ("RK45", False),
        ("RK45", True),
        ("DOP853", False),
        ("DOP853", True),
    ],
)
def test_solver_method_combinations(solver_method, dense_output):
    """様々な ODE 解法の組み合わせで固有対追跡が成功することを確認する。"""
    t_span = (0.0, 1.0)
    t_eval = np.linspace(*t_span, 5)

    kwargs = {"dense_output": dense_output}
    if solver_method is not None:
        kwargs["solver_method"] = solver_method

    results = eigenpairtrack(
        _basic_matrix,
        _d_basic_matrix,
        t_span,
        t_eval,
        **kwargs,
    )

    assert results.success
    assert len(results.t_eval) == len(t_eval)


def test_euler_requires_initial_time():
    """Euler 法では ``t_eval`` の最初の値が開始時刻に一致しない場合に例外を送出することを確認する。"""
    t_span = (0.0, 1.0)
    t_eval = np.linspace(0.1, 1.0, 5)

    with pytest.raises(ValueError):
        eigenpairtrack(
            _basic_matrix,
            _d_basic_matrix,
            t_span,
            t_eval,
            solver_method="Euler",
        )
