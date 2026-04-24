import numpy as np
import scipy.linalg as lg
import scipy.optimize as opt

from .Logger import LOG
from ._optimize import quiet_newton_krylov

from typing import Callable, List, Tuple, Dict

def _solveABSystem(a, b, c):
    """
    Simple function that solves the quadratic form 
        a alpha**2 + 2 b alpha beta + c beta**2 = 0.
    
    Parameters
    ----------
    a, b, c : float
        The three homogeneous coefficients.
    
    Returns
    -------
        solutions : List[ndarray]
            The (typically four) unit-vector solutions to the homogeneous quadratic system.
    """
    scale = max(abs(a), abs(b), abs(c), 1.0)
    tol = 1e-12 * scale
    solutions: List[np.ndarray] = []

    def add_solution(alpha: float, beta: float) -> None:
        x = np.array([alpha, beta], dtype=float)
        norm_x = lg.norm(x)
        if norm_x <= tol or not np.isfinite(norm_x):
            return
        x /= norm_x
        if not any(lg.norm(x - y) <= 1e-10 for y in solutions):
            solutions.append(x)

    # Degenerate quadratic: every unit direction is a solution. Returning four
    # canonical directions preserves the expected branch-switching shape.
    if abs(a) <= tol and abs(b) <= tol and abs(c) <= tol:
        return [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]

    discriminant = b*b - a*c
    if discriminant < -tol:
        return solutions
    discriminant = max(0.0, discriminant)
    sqrt_discriminant = np.sqrt(discriminant)

    if abs(c) > tol:
        for beta_over_alpha in ((-b + sqrt_discriminant) / c, (-b - sqrt_discriminant) / c):
            add_solution(1.0, beta_over_alpha)
            add_solution(-1.0, -beta_over_alpha)
    elif abs(b) > tol:
        beta_over_alpha = -a / (2.0*b)
        add_solution(1.0, beta_over_alpha)
        add_solution(-1.0, -beta_over_alpha)
        add_solution(0.0, 1.0)
        add_solution(0.0, -1.0)
    else:
        add_solution(0.0, 1.0)
        add_solution(0.0, -1.0)

    return solutions

# Gu_v takes arguments u, p, v
def _computeCoefficients(Gu_v : Callable[[np.ndarray, float, np.ndarray], np.ndarray], 
                         Gp : Callable[[np.ndarray, float], np.ndarray], 
                         x_singular : np.ndarray, 
                         phi : np.ndarray, 
                         w : np.ndarray, 
                         w_1 : np.ndarray, 
                         M : int, 
                         r_diff : float) -> Tuple[float, float, float]:
    """
    Compute the coefficients a, b, c in the quadratic form 
        alpha**2 a + 2 alpha beta b + beta**2 c.
    See equation () from [] for more details.

    Parameters
    ----------
    Gu : Callable
        Jacobian of G at the bifurcation point. Callable of signature `Gu(v)` where
        `v` is the direction in which to compute the directional derivative
    Gp : ndarray
        Derivative of `G` with respect to the parameter `p` at the bifurcation point.
    x_singular : ndarray
        The bifurcation point
    phi : ndarray
        Unit vector in the nullspace of Gu(x_singular).
    w : ndarray
        Unit vector that solves Gu * w + Gp = 0.
    w_1 : ndarray
        Unit vector in the nullspace of [Gu | Gp]
    M : int
        Size of the state vector u.
    r_diff : float
        Finite-differences step size for the second-order derivatives

    Returns
    -------
    a, b, c : float
        The three second-order coefficients.
    """

    phi_exp = np.append(phi, 0.0)

    # Compute a
    Gu_phi = lambda x: Gu_v(x[0:M], x[M], phi)
    a = np.dot(phi, (Gu_phi(x_singular + r_diff * phi_exp) - Gu_phi(x_singular)) / r_diff)

    # Compute b
    Gx_w = lambda x: Gu_v(x[0:M], x[M], w) + Gp(x[0:M], x[M])
    b = np.dot(phi, (Gx_w(x_singular + r_diff * phi_exp) - Gx_w(x_singular)) / r_diff)

    # Compute c
    c = np.dot(phi, (Gx_w(x_singular + r_diff * w_1) - Gx_w(x_singular)) / r_diff)

    return a, b, c

def _computeNullspace(Gu : Callable[[np.ndarray], np.ndarray], 
                      Gp : np.ndarray, 
                      M : int, 
                      r_diff : float):
    """
    Compute the nullspaces of the Jacobian Gu and of the extended matrix [Gu | Gp].

    Parameters
    ----------
    Gu : Callable
        Jacobian of G at the bifurcation point. Callable of signature `Gu(v)` where
        `v` is the direction in which to compute the directional derivative
    Gp : ndarray
        Derivative of `G` with respect to the parameter `p` at the bifurcation point.
    M : int
        Dimension of the state vector `u`
    r_diff : float
        Finite-differences step size for minimization routine.

    Returns
    -------
    phi : ndarray
        Unit nullvector of Gu
    w : ndarray
        Vector that solves Gu * w + Gp = 0
    w_1 : ndarray
        Unit nullvector of [Gu | Gp] obtained by appending a 1 to w.
    """
    phi_0 = np.eye(M)[:,0]
    def phi_residual(y: np.ndarray) -> np.ndarray:
        return np.append(Gu(y), np.dot(y, y) - 1.0)
    min_result = opt.least_squares(phi_residual, phi_0)
    phi = min_result.x
    phi_norm = lg.norm(phi)
    if phi_norm > 0.0:
        phi = phi / phi_norm

    try:
        w = quiet_newton_krylov(lambda y: Gu(y) + Gp, np.zeros(M), rdiff=r_diff)
    except opt.NoConvergence as e:
        w = e.args[0]
    w_1 = np.append(w, 1.0)
    w_1 = w_1 / lg.norm(w_1)

    return phi, w, w_1

def branchSwitching(G : Callable[[np.ndarray, float], np.ndarray],  
                    x_singular : np.ndarray, 
                    x_prev : np.ndarray, 
                    sp : Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform branch switching at the current bifurcation point `x_singular'.

    Parameters
    ----------
    G : callable
        The objective function.
    x_singular : ndarray
        The bifurcation point (sometimes called a `singular point').
    x_prev : ndarray
        Final point along the branch before the bifurcation point.
    sp : Dict
        Solver parameters.

    Returns
    -------
    directions : List[ndarray]
        List containing the starting points of the new branches, far enough
        away from x_singular.
    tangents : List[ndarray]
        Tangent vectors at the new branches in the starting points.
    """
    LOG.info('\nBranch Switching')

    # Setting up variables
    M = x_singular.size - 1
    u = x_singular[0:M]
    p = x_singular[M]

    # Create gradient functions
    rdiff = sp["rdiff"]
    def Gu_v(u : np.ndarray, p : float, v : np.ndarray) -> np.ndarray:
        norm_v = lg.norm(v)
        if norm_v == 0.:
            return np.zeros_like(u)
        eps = rdiff / norm_v
        return (G(u + eps * v, p) - G(u - eps * v, p)) / (2.0 * eps)
    Gp = lambda u, p: (G(u, p + rdiff) - G(u, p - rdiff)) / (2.0 * rdiff)

    # Computing necessary coefficients and vectors
    phi, w, w_1 =_computeNullspace(lambda v: Gu_v(u, p, v), Gp(u,p), M, rdiff)
    a, b, c = _computeCoefficients(Gu_v, Gp, x_singular, phi, w, w_1, M, rdiff)
    solutions = _solveABSystem(a, b, c)

    # Fina all 4 branch tangents
    directions = []
    tangents = []
    for n in range(len(solutions)):
        alpha = solutions[n][0]
        beta  = solutions[n][1]

        #s = 0.01
        branch_u = alpha*phi + beta*w
        branch_p = beta
        N = lambda x: np.dot(branch_u, x[0:M] - x_singular[0:M]) + branch_p*(x[M] - x_singular[M]) - sp["s_jump"]
        F_branch = lambda x: np.append(G(x[0:M], x[M]), N(x))

        tangent = np.append(branch_u, branch_p)
        x0 = x_singular + sp["s_jump"] * tangent / lg.norm(tangent)
        dir = quiet_newton_krylov(F_branch, x0, rdiff=sp["rdiff"], f_tol=sp["tolerance"])

        directions.append(dir)
        tangents.append(tangent)

    if not directions:
        return directions, tangents

    # Remove the direction where we came from
    inner_prodct = -np.inf
    for n in range(len(directions)):
        inner_pd = np.dot(directions[n]-x_singular, x_prev-x_singular) / (lg.norm(directions[n]-x_singular) * lg.norm(x_prev-x_singular))
        if inner_pd > inner_prodct:
            inner_prodct = inner_pd
            idx = n
    directions.pop(idx)
    tangents.pop(idx)
    LOG.info(lambda: f'Branch Switching Tangents: {tangents}')

    # Returning 3 continuation directions
    return directions, tangents
