import pathlib
import sys
import unittest
from unittest import mock

import numpy as np
import scipy.optimize as opt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pycont.detection.parammax import ParamMaxDetectionModule
from pycont.detection.parammin import ParamMinDetectionModule


class ParamBoundRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sp = {"rdiff": 1e-6, "tolerance": 1e-8}

    @staticmethod
    def G(u: np.ndarray, p: float) -> np.ndarray:
        return u - p

    def test_param_max_localize_falls_back_to_interpolation_on_value_error(self) -> None:
        module = ParamMaxDetectionModule(self.G, np.array([0.0]), 0.0, self.sp, param_max_value=0.5)
        module.initializeBranch(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        self.assertTrue(module.update(lambda x: x, np.array([1.0, 1.0]), np.array([1.0, 1.0])))

        with mock.patch("pycont.detection._parambound.quiet_newton_krylov", side_effect=ValueError("array must not contain infs or NaNs")):
            x_param_max = module.localize()

        np.testing.assert_allclose(x_param_max, np.array([0.5, 0.5]))

    def test_param_max_localize_falls_back_to_interpolation_on_zero_division(self) -> None:
        def shifted_G(u: np.ndarray, p: float) -> np.ndarray:
            return u - p + 1e-3

        module = ParamMaxDetectionModule(shifted_G, np.array([0.0]), 0.0, self.sp, param_max_value=0.5)
        module.initializeBranch(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        self.assertTrue(module.update(lambda x: x, np.array([1.0, 1.0]), np.array([1.0, 1.0])))

        with mock.patch("pycont.detection._parambound.quiet_newton_krylov", side_effect=ZeroDivisionError("float division by zero")):
            x_param_max = module.localize()

        np.testing.assert_allclose(x_param_max, np.array([0.5, 0.5]))

    def test_param_min_localize_falls_back_when_no_convergence_iterate_is_non_finite(self) -> None:
        module = ParamMinDetectionModule(self.G, np.array([1.0]), 1.0, self.sp, param_min_value=0.5)
        module.initializeBranch(np.array([1.0, 1.0]), np.array([1.0, -1.0]))
        self.assertTrue(module.update(lambda x: x, np.array([0.0, 0.0]), np.array([1.0, -1.0])))

        failure = opt.NoConvergence(np.array([np.nan]))
        with mock.patch("pycont.detection._parambound.quiet_newton_krylov", side_effect=failure):
            x_param_min = module.localize()

        np.testing.assert_allclose(x_param_min, np.array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
