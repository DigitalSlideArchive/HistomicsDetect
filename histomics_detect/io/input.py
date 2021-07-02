import pandas as pd
import tensorflow as tf


def read_roi(roi):
    """Reads the input image from disk and stacks ground truth boxes
    in a ragged tensor.
    
    Ragged tensors are needed because the number of boxes associated
    with different images is variable. Input pipelines require uniform
    sizes for standard tensors, where ragged tensors allow these functions
    to be used in input pipelines when processing variable length data.
        
    Parameters
    ----------
    roi: dict
        This dict contains the path of the image file ('png') and the 
        positions ('x', 'y') and size ('width', 'height') of associated
        bounding boxes.
        
    Returns
    -------
    rgb: tensor
        The image as a 2D or 3D tensor.
    boxes: tensor (float32)
        N x 4 tensor containing boxes where each row contains the x,y 
        location of the upper left corner of a ground truth box and its width 
        and height in that order.
    """
    
    #read in png and get size
    rgb = tf.io.decode_png(tf.io.read_file(roi['png'])) 

    #get roi dimensions
    height = tf.shape(rgb)[0]
    width = tf.shape(rgb)[1]

    #stack into ragged tensor
    boxes = tf.RaggedTensor.from_tensor(tf.stack((roi['x'], roi['y'],
                                                  roi['width'], roi['height']),
                                                 axis=1))

    return rgb, boxes, roi['png']


def roi_tensors(files):
    """Generates dictionary of image file path and ground truth box values.
    
    Uses pandas to read csv files containing ground truth box information
    and links these with the associated image filenames in a dict for use
    with tensorflow Datasets.
        
    Parameters
    ----------
    files: array_like
        A list of (.png, .csv) tuples containing the filenames and paths
        to image files and associated bounding box tables.
        
    Returns
    -------
    roi: dict
        This dict contains the path of the image file ('png') and the 
        positions ('x', 'y') and size ('width', 'height') of associated
        bounding boxes.
    """

    #parse .csv files to dict for tf dataset
    rois = {'png': [], 'x': [], 'y': [], 'width': [], 'height': []}
    for i, (png, csv) in enumerate(files):

        #read .csv
        table = pd.read_csv(csv)

        #extract x, y, width, height
        table = table.loc[table['type'] == 'box']
        rois['png'].append(tf.convert_to_tensor(png))
        rois['x'].append(tf.constant(table['x'], tf.float32))
        rois['y'].append(tf.constant(table['y'], tf.float32))
        rois['width'].append(tf.constant(table['width'], tf.float32))
        rois['height'].append(tf.constant(table['height'], tf.float32))

    rois['png'] = tf.ragged.stack(rois['png'])    
    rois['x'] = tf.ragged.stack(rois['x'])
    rois['y'] = tf.ragged.stack(rois['y'])
    rois['width'] = tf.ragged.stack(rois['width'])
    rois['height'] = tf.ragged.stack(rois['height'])

    return rois
