"""
The networks package contains functions for creating and manipulating networks
"""

# make functions available at the package level using shadow imports
from .backbones import residual_backbone
from .fast_rcnn import fast_rcnn
from .field_size import field_size
from .rpns import rpn
from .transfer_layers import transfer_layers

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'residual_backbone',
    'fast_rcnn',
    'field_size',
    'rpn',
    'transfer_layers'
)
