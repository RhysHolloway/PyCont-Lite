import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pycont.detection._hopf import _JacobiDavidson
from pycont.detection.hopf import HopfDetectionModule, HopfState


class HopfRobustnessTests(unittest.TestCase):
    def test_jacobi_davidson_keeps_previous_pair_when_residual_is_non_finite(self) -> None:
        lam0 = np.complex128(0.5 + 0.25j)
        v0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

        def bad_jacobian_vector_product(v: np.ndarray) -> np.ndarray:
            return np.array([np.nan + 0.0j, 1.0 + 0.0j], dtype=np.complex128)

        lam, v = _JacobiDavidson(bad_jacobian_vector_product, lam0, v0, tolerance="weak")

        self.assertEqual(lam, lam0)
        np.testing.assert_allclose(v, v0)

    def test_confident_hopf_state_requires_a_tracked_complex_pair(self) -> None:
        state = HopfState(
            x=np.zeros(3),
            eigvals=np.array([1.0 + 0.0j], dtype=np.complex128),
            eigvecs=np.ones((1, 1), dtype=np.complex128),
            lead=-1,
        )

        self.assertFalse(HopfDetectionModule._is_confident_state(state))


if __name__ == "__main__":
    unittest.main()
