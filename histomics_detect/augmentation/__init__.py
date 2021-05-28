"""
This package contains functions for data augmentation
"""

# make functions available at the package level using shadow imports
from .augmentation import flip
from .augmentation import crop
from .augmentation import box_crop

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'flip',
    'crop',
    'box_crop'
)
