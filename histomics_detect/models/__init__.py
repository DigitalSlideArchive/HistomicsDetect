"""
This package contains Keras Model subclasses for detection
"""

# make functions available at the package level using shadow imports
from .faster_rcnn import FasterRCNN
from .faster_rcnn import CustomCallback

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'FasterRCNN',
    'CustomCallback'
)
