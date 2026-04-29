import numpy as np
import scipy.optimize as opt

from typing import Any, Callable

def quiet_newton_krylov(F: Callable[[Any], Any], xin: Any, *args: Any, **kwargs: Any) -> Any:
    """Call SciPy's Newton-Krylov solver without emitting iteration traces."""
    kwargs["verbose"] = False
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        try:
            residual_norm = np.linalg.norm(np.asarray(F(xin)))
        except Exception:
            residual_norm = np.inf
    if np.isfinite(residual_norm) and residual_norm == 0.0:
        return xin
    return opt.newton_krylov(F, xin, *args, **kwargs)
