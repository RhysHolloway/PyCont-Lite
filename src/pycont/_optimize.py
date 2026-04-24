import scipy.optimize as opt

from typing import Any, Callable

def quiet_newton_krylov(F: Callable[[Any], Any], xin: Any, *args: Any, **kwargs: Any) -> Any:
    """Call SciPy's Newton-Krylov solver without emitting iteration traces."""
    kwargs["verbose"] = False
    return opt.newton_krylov(F, xin, *args, **kwargs)
