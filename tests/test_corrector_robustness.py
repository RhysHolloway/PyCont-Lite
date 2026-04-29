import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pycont import ArclengthContinuation as pac


class CorrectorRobustnessTests(unittest.TestCase):
    def test_corrector_halves_step_after_non_finite_krylov_failure(self) -> None:
        def G(u: np.ndarray, p: float) -> np.ndarray:
            return u - p

        ds = 0.08
        ds_after_retry = 0.5 * ds
        tangent = np.array([1.0, 1.0]) / np.sqrt(2.0)
        x_retry = tangent * ds_after_retry
        sp = {"nk_maxiter": 10, "rdiff": 1e-6, "tolerance": 1e-10}

        with mock.patch(
            "pycont.ArclengthContinuation.quiet_newton_krylov",
            side_effect=[ValueError("array must not contain infs or NaNs"), x_retry],
        ) as newton_krylov:
            branch, event = pac.continuation(
                G,
                np.array([0.0]),
                0.0,
                tangent,
                0.01,
                0.1,
                ds,
                1,
                0,
                [],
                sp,
            )

        self.assertEqual(newton_krylov.call_count, 2)
        self.assertEqual(event.kind, "MAXSTEPS")
        np.testing.assert_allclose(branch.u_path[-1], np.array([x_retry[0]]))
        self.assertAlmostEqual(branch.p_path[-1], x_retry[1])
        self.assertAlmostEqual(branch.s_path[-1], ds_after_retry)


if __name__ == "__main__":
    unittest.main()
