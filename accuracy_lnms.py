#import dataset related packages
# from histomics_detect.io import roi_tensors
# from histomics_detect.io import read_roi
from histomics_detect.augmentation import crop
from histomics_detect.augmentation import flip
# from histomics_detect.augmentation.augmentation import jitter_boxes
from histomics_detect.boxes.transforms import filter_edge_boxes
import numpy as np
import os
from PIL import Image
from histomics_detect.visualization.visualization import _plot_boxes

import tensorflow as tf
import matplotlib.pyplot as plt

#input data path
path = '../DCC/'

#training parameters
train_tile = 224 #input image size
min_area_thresh = 0.5 # % of object area that must be in crop to be included

#build (.png, image size, slide, .csv) file tuples
files = os.listdir(path)
pngs = [path + f for f in files if f.split('.')[-1] == 'png']
path_len = len(path)
files = [(png, Image.open(png).size, png[path_len:].split('.')[2],
          '.'.join(png.split('.')[0:-1]) + '.csv') for png in pngs]

#filter on png size
files = [(png, size, slide, csv) for (png, size, slide, csv) in files
         if (size[0] > train_tile) and (size[1] > train_tile)]

#randomly assign 20% of slides to validation
slides = list(set([file[2] for file in files]))
id = np.random.randint(0, len(slides)-1, size=(np.ceil(0.2*len(slides)).astype(np.int32)))
validation = [slide for (i, slide) in enumerate(slides) if i in id]
training = list(set(slides).difference(validation))
training_files = [(png, csv) for (png, size, slide, csv) in files if slide in training]
validation_files = [(png, csv) for (png, size, slide, csv) in files if slide in validation]

#convert to tensors
training_rois = roi_tensors(training_files)
validation_rois = roi_tensors(validation_files)

#arguments
width = tf.constant(train_tile, tf.int32)
height = tf.constant(train_tile, tf.int32)
min_area = tf.constant(min_area_thresh, tf.float32)

#build training dataset
ds_train_roi = tf.data.Dataset.from_tensor_slices(training_rois)
ds_train_roi = ds_train_roi.map(lambda x: read_roi(x))
ds_train_roi = ds_train_roi.map(lambda x, y: crop(x, y, width, height,
                                                            min_area_thresh))
ds_train_roi = ds_train_roi.map(lambda x, y: flip(x, y))
# ds_train_roi = ds_train_roi.map(lambda x, y: (x, jitter_boxes(y, 0.5, 'yx'), y))

ds_train_roi = ds_train_roi.prefetch(tf.data.experimental.AUTOTUNE)

#build validation datasets
ds_validation_roi = tf.data.Dataset.from_tensor_slices(validation_rois)
ds_validation_roi = ds_validation_roi.map(lambda x: read_roi(x))
ds_validation_roi = ds_validation_roi.prefetch(tf.data.experimental.AUTOTUNE)

from typing import Tuple
from histomics_detect.metrics import iou


def calculate_performance_stats(boxes: tf.Tensor, rpn_boxes: tf.Tensor, scores: tf.Tensor) -> Tuple[int, int, int, int]:

    ious = iou(boxes, rpn_boxes_positive)

    def func(i) -> tf.int32:
        index = tf.cast(i, tf.int32)
        assignment = tf.cast(tf.argmax(ious[index]), tf.int32)
        return assignment

    indeces = tf.expand_dims(tf.map_fn(lambda x: func(x), tf.range(0, tf.shape(ious)[0])), axis=1)
    labels = tf.scatter_nd(indeces, tf.ones(tf.shape(indeces)), tf.shape(scores))
    print(indeces)

    tp = tf.reduce_sum(tf.cast(tf.logical_and(labels == 1, scores > 0.5), tf.int32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(labels == 0, scores <= 0.5), tf.int32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(labels == 0, scores > 0.5), tf.int32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(labels == 1, scores <= 0.5), tf.int32))

    return tp, tn, fp, fn

samples = 2
variance = [4, 4, 6, 6]

for data in ds_train_roi:
    rgb, boxes = data

    boxes = boxes.to_tensor()
    rpn_boxes_positive = tf.tile(boxes, [samples, 1]) + tf.stack(
        [tf.random.normal([tf.shape(boxes)[0] * samples], stddev=x)
         for x in variance], axis=1)

    rpn_boxes_positive = tf.concat([boxes + tf.random.normal(tf.shape(boxes), stddev=1), rpn_boxes_positive], axis=0)
    rpn_boxes_positive = tf.stack([rpn_boxes_positive[:, 0], rpn_boxes_positive[:, 1],
                                   tf.math.abs(rpn_boxes_positive[:, 2]) + 1,
                                   tf.math.abs(rpn_boxes_positive[:, 3]) + 1], axis=1)

    fig = plt.figure()
    plt.imshow(rgb)
    _plot_boxes(boxes, 'r')
    _plot_boxes(rpn_boxes_positive, 'b')

    plt.show()

    scores = tf.ones((tf.shape(rpn_boxes_positive)[0], 1))
    tp, tn, fp, fn = calculate_performance_stats(boxes, rpn_boxes_positive, scores)
    print("boxes shape", tf.shape(boxes))
    print("rpn boxes shape", tf.shape(rpn_boxes_positive))
    print("tp tn fp fn", tp, tn, fp, fn)
    break