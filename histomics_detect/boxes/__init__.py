"""
This package contains functions for modifying bounding boxes
"""

# make functions available at the package level using shadow imports
from .transforms import clip_boxes
from .transforms import tf_box_transform
from .transforms import parameterize
from .transforms import unparameterize
from .transforms import filter_edge_boxes
from .neighborhood import assemble_single_neighborhood, all_neighborhoods_additional_info
from .match import cluster_assignment

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'clip_boxes',
    'tf_box_transform',
    'parameterize',
    'unparameterize',
    'filter_edge_boxes',
    'assemble_single_neighborhood',
    'all_neighborhoods_additional_info',
    'cluster_assignment'
)
