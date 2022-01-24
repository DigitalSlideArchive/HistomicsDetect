import os
from PIL import Image
import tensorflow as tf


def dataset(path, png_parser, csv_parser, size, cases=None):
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
        An optional list of cases used to select rois for inclusion in the 
        dataset. Default value of None will select all cases in 'path'.
        Helpful when cases are grouped by folder
        
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
    if cases is not None:
        matches = [match for match in matches if match[0] in cases]
            
    #format outputs
    matches = [(path + match[1], path + match[2]) for match in matches]
    
    #filter on image size
    matches = [(png, csv) for (png, csv) in matches if
               (Image.open(png).size[0] > size) and (Image.open(png).size[1] > size)]
    
    #create tf.data.Dataset object of pairs of csv/png files
    ds = tf.data.Dataset.from_tensor_slices(matches)
    
    #map image and csv read operations to generate (rgb, boxes, pngfile) tuple
    ds = ds.map(lambda x: (read_png(x[0]), read_csv(x[1]), x[0]))
    
    return ds


@tf.function
def read_csv(csv_file):
    """
    Reads a csv file describing bounding boxes using tensorflow operations.
    
    Each file should include a single header row, and each subsequent rows
    describes a single box and contains the fields
    
        x (float) - horizontal upper-left corner of box (pixels)
        y (float) - vertical upper-left corner of box (pixels)
        w (float) - box width (pixels)
        h (float) - box height (pixels)
        label (string) - class label (optional)
        center_x (float - optional) - horizontal center of box (for point annotations)
        center_y (float - optional) - vertical center of box (for point annotations)
        slide_x (float - optional) - horizontal upper-left corner of box in slide
        slide_y (float - optional) - vertical upper-left corner of box in slide
        type (string - optional) - the type of annotation, either 'box' or 'point'
        contained (bool - optional) - if box or point is entirelycontained within roi
        
    Parameters
    ----------
    csv_file: string
        Path and filename of the csv file.
        
    Returns
    -------
    ds: RaggedTensor
        A ragged tensor where the each row contains the x,y location 
        of the upper left corner of a ground truth box and its width and
        height in that order.
    """
    
    #read contents of csv
    contents = tf.io.read_file(csv_file)

    #split into lines
    lines = tf.strings.split(contents, '\n')

    #decode data lines
    lines = tf.io.decode_csv(lines[1:-1],
                             [0.0, 0.0, 0.0, 0.0, '', 0.0, 0.0, 0.0, 0.0, '', ''])
    
    #embed in ragged tensor
    boxes = tf.RaggedTensor.from_tensor(tf.stack((lines[0], lines[1], 
                                                  lines[2], lines[3]), 
                                                 axis=1))

    return boxes


@tf.function
def read_png(png_file):
    """
    Reads a png file using tensorflow operations.
        
    Parameters
    ----------
    png_file: string
        Path to the png file.
        
    Returns
    -------
    rgb: tensor
        Three dimensional rgb image tensor.
    """
    
    #read in png and get size
    rgb = tf.io.decode_png(tf.io.read_file(png_file))
    
    return rgb


@tf.function
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
