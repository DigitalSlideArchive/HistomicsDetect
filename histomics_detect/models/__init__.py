"""
This package contains Keras Model subclasses for detection
"""

# make functions available at the package level using shadow imports
from .faster_rcnn import FasterRCNN
from .lnms_loss import normal_loss, xor_loss, clustering_loss, paper_loss
from .compression_network import CompressionNetwork
from .lnms_model import LearningNMS
from .block_model import BlockModel
from .model_utils import extract_data
# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'FasterRCNN',
    'normal_loss',
    'xor_loss',
    'clustering_loss',
    'paper_loss',
    'CompressionNetwork',
    'LearningNMS',
    'BlockModel',
    'extract_data'
)
