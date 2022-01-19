"""
This package contains functions for visualization of detection results
"""

# make functions available at the package level using shadow imports
from .visualization import plot_evaluation
from .visualization import plot_inference
from .lnms_visualization import plot_inference as plot_inference_lnms, run_plot, plot_multiple_outputs

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'plot_evaluation',
    'plot_inference',
    'plot_inference_lnms',
    'run_plot',
    'plot_multiple_outputs'
)
