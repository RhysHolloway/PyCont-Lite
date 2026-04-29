from .continuation import pseudoArclengthContinuation as arclengthContinuation
from .Logger import Verbosity

__version__ = "0.6.0"

def plotBifurcationDiagram(*args, **kwargs):
    from .plotting import plotBifurcationDiagram as _plotBifurcationDiagram

    return _plotBifurcationDiagram(*args, **kwargs)

__all__ = [
    "arclengthContinuation", 
    "plotBifurcationDiagram",
    "Verbosity",
    "__version__",
]
