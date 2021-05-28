import numpy as np


def field_size(backbone, length=65):
    """Estimates receptive field size of the backbone network.
    
    The receptive field size is estimated empirically by passing arrays of increasing
    sizes through the backbone and comparing the ratio of sizes of the input and output.
    The max of this sequence of ratios corresponds to the case where the field size
    evenly divides the input with no remainder.
        
    Parameters
    ----------
    backbone: keras Model
        The backbone network as a keras Model.
    length: int32
        Maximum size of input to use in estimating field size.
        
    Returns
    -------
    field: float32
        Field size of the backbone network in pixels.
    """    
  
    ratio = []
    for dim in np.arange(1, length, 1):
        im = np.zeros(shape=(dim, dim, 3), dtype=np.uint8)
        prediction = backbone.predict(np.array([im,])).shape
        ratio.append(dim/prediction[2])
    field = np.max(ratio)

    return field
