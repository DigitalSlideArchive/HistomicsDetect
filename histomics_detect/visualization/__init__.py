"""
This package contains functions for visualization of detection results
"""

# make functions available at the package level using shadow imports
from .visualization import plot_evaluation
from .visualization import plot_inference

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'plot_evaluation',
    'plot_inference'
)
