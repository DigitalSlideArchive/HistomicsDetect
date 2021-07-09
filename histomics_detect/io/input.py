import os
import pandas as pd
from PIL import Image
import tensorflow as tf


def dataset(path, png_parser, csv_parser, size, cases):
    """Generates a tf.data.Dataset object containing matched region
    of interest .pngs and bounding box .csv files.
    
    This function accepts parser function handles that are used to parse
    and match png and csv filenames. Each parser is written to return 
    case and unique roi identifier strings that are used for matching
    png and csv files and for filtering cases from the match list.
        
    Parameters
    ----------
    path: string
        Path containing png region of interest images and csv files 
        containing corresponding bounding boxes.
    png_parser: function
        Function that accepts a single string containing the filename
        of a png image (without path) and that returns the corresponding
        case and roi name. The case and roi name should uniquely identify
        the roi within the dataset.
    csv_parser: function
        Function that accepts a single string containing the filename
        of a png image (without path) and that returns the corresponding
        case and roi name. The case and roi name should uniquely identify
        the roi within the dataset.
    cases: list of strings
        A list of cases used to select rois for inclusion in the 
        dataset.
        
    Returns
    -------
    ds: tf.data.Dataset
        A dataset where each element contains an rgb image tensor, an N x 4
        tensor of bounding boxes where each row contains the x,y location 
        of the upper left corner of a ground truth box and its width and
        height in that order, and the png filename.
    """
    
    #get list of csv and png files in path   
    csvs = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.csv']
    pngs = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.png']
        
    #extract case, roi strings from filenames
    csv_case_roi = [csv_parser(csv) for csv in csvs]
    png_case_roi = [png_parser(png) for png in pngs]
        
    #form lists of case + roi for matching
    csv_match_string = [csv[0] + csv[1] for csv in csv_case_roi]
    png_match_string = [png[0] + png[1] for png in png_case_roi]
        
    #match
    indexes = [csv_match_string.index(png) if png in csv_match_string else -1 for 
               png in png_match_string]
    
    #form tuples of case, matching png file, csv file
    matches = [(case_roi[0], png, csvs[index]) for (case_roi, png, index) in 
               zip(png_case_roi, pngs, indexes) if index != -1]
        
    #filter on cases
    matches = [match for match in matches if match[0] in cases]
            
    #format outputs
    matches = [(path + match[1], path + match[2]) for match in matches]
    
    #filter on image size
    matches = [(png, csv) for (png, csv) in matches if
               (Image.open(png).size[0] > size) and (Image.open(png).size[1] > size)]    
    
    #build dataset
    ds = tf.data.Dataset.from_tensor_slices(roi_tensors(matches))
    ds = ds.map(lambda x: read_roi(x))
    
    return ds


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


def resize(rgb, boxes, factor):
    """Resizes input image and bounding boxes.
    
    This function can be used as a transformation in the input pipeline to 
    resize the images and bounding boxes. This can help to improve performance
    in images with small objects or images where object density is high
    relative to anchor density.
        
    Parameters
    ----------
    rgb: tensor
        The image as a 2D or 3D tensor.
    boxes: tensor (float32)
        N x 4 tensor containing boxes where each row contains the x,y 
        location of the upper left corner of a ground truth box and its width 
        and height in that order.
    factor: float32
        The scalar factor used for resizing. A value of 2.0 will double the
        magnification.
        
    Returns
    -------
    rgb: tensor
        The input image resized by factor.
    boxes: tensor (float32)
        N x 4 tensor containing boxes resized by factor.
    """
    
    #resize image
    rgb = tf.image.resize(rgb, tf.cast([factor * tf.cast(tf.shape(rgb)[0], tf.float32), 
                                        factor * tf.cast(tf.shape(rgb)[1], tf.float32)], 
                                       tf.int32))
    
    #resize boxes
    boxes = factor * boxes
    
    return rgb, boxes


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
