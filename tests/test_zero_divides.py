import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pycont import arclengthContinuation
from pycont.LimitCycle import calculateInitialLimitCycle, createLimitCycleObjectiveFunction
from pycont.detection._bifurcation import test_fn_jacobian
from pycont.exceptions import InputError


class ZeroDivideGuardsTests(unittest.TestCase):
    def test_continuation_rejects_zero_rdiff(self) -> None:
        def G(u: np.ndarray, p: float) -> np.ndarray:
            return u - p

        with self.assertRaises(InputError):
            arclengthContinuation(
                G,
                np.array([0.0]),
                0.0,
                1e-6,
                1e-2,
                1e-3,
                10,
                {"rdiff": 0.0},
                verbosity="off",
            )

    def test_bifurcation_test_function_returns_inf_when_denominator_vanishes(self) -> None:
        sp = {"rdiff": 1e-6}
        F = lambda x: np.zeros_like(x)
        x = np.zeros(2)
        l = np.array([1.0, 1.0])
        r = np.zeros(2)
        w_prev = np.zeros(2)
        w_solution = np.array([1.0, -1.0])

        with mock.patch("pycont.detection._bifurcation.quiet_newton_krylov", return_value=w_solution):
            _, beta = test_fn_jacobian(F, x, l, r, w_prev, sp)

        self.assertTrue(np.isinf(beta))

    def test_limit_cycle_initialization_rejects_zero_frequency(self) -> None:
        def G(u: np.ndarray, p: float) -> np.ndarray:
            return u

        lc = calculateInitialLimitCycle(
            G,
            {"rdiff": 1e-6},
            np.array([0.0, 0.0, 0.0]),
            0.0,
            np.array([1.0 + 0.0j, 0.0 + 1.0j]),
            2,
        )

        self.assertIsNone(lc)

    def test_limit_cycle_objective_rejects_non_positive_collocation_count(self) -> None:
        def G(u: np.ndarray, p: float) -> np.ndarray:
            return u

        with self.assertRaises(ValueError):
            createLimitCycleObjectiveFunction(G, np.zeros(2), 1, L=0)


if __name__ == "__main__":
    unittest.main()
