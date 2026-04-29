import numpy as np
import scipy.optimize as opt

from ..Logger import LOG
from .._optimize import quiet_newton_krylov

from typing import Any, Callable, Dict


def _all_finite(x: np.ndarray | float) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


def localizeParameterBoundary(
    G: Callable[[np.ndarray, float], np.ndarray],
    u_prev: np.ndarray,
    p_prev: float,
    u_new: np.ndarray,
    p_new: float,
    target_p: float,
    sp: Dict[str, Any],
    label: str,
) -> np.ndarray:
    """
    Localize a user-requested parameter boundary.

    We first interpolate linearly between the accepted continuation points and then
    optionally refine at fixed `target_p` using Newton-Krylov. If the refinement path
    becomes numerically non-finite, keep the interpolated boundary point instead of
    aborting the whole continuation run.
    """
    if p_new == p_prev:
        alpha = 0.0
    else:
        alpha = (target_p - p_prev) / (p_new - p_prev)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    u_guess = np.array(u_prev + alpha * (u_new - u_prev), copy=True)

    objective = lambda u: G(u, target_p)
    rdiff = sp["rdiff"]
    tolerance = sp["tolerance"]

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        try:
            residual_guess = objective(u_guess)
        except Exception:
            residual_guess = None

    if not _all_finite(u_guess):
        LOG.info(lambda: f"{label} localization produced a non-finite interpolated state. Using the last finite boundary estimate.")
        return u_guess

    if residual_guess is not None and _all_finite(residual_guess):
        residual_norm = np.linalg.norm(residual_guess)
        LOG.verbose(lambda: f"{label} interpolation residual {residual_norm}")
        if np.isfinite(residual_norm) and residual_norm <= 10.0 * tolerance:
            return u_guess
    else:
        LOG.info(lambda: f"{label} localization objective is non-finite at the interpolated boundary state. Skipping Newton refinement.")
        return u_guess

    try:
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            u_refined = quiet_newton_krylov(objective, u_guess, rdiff=rdiff, f_tol=tolerance)
    except opt.NoConvergence as e:
        u_refined = e.args[0]
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError) as e:
        LOG.info(lambda: f"{label} localization refinement failed ({type(e).__name__}). Using interpolated boundary state instead.")
        return u_guess

    if not _all_finite(u_refined):
        LOG.info(lambda: f"{label} localization produced a non-finite refined state. Using interpolated boundary state instead.")
        return u_guess

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        try:
            residual_refined = objective(u_refined)
        except Exception:
            residual_refined = None

    if residual_refined is None or not _all_finite(residual_refined):
        LOG.info(lambda: f"{label} localization residual became non-finite after refinement. Using interpolated boundary state instead.")
        return u_guess

    residual_norm = np.linalg.norm(residual_refined)
    LOG.verbose(lambda: f"{label} refined residual {residual_norm}")
    if not np.isfinite(residual_norm):
        LOG.info(lambda: f"{label} localization residual norm is non-finite after refinement. Using interpolated boundary state instead.")
        return u_guess

    return np.array(u_refined, copy=True)
