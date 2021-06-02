================================================
histomics_detect
================================================

`histomics_detect`_ is a Python package for the building and evaluating cell detection 
models. It provides data loading, data augmentation, performance metrics, model building,
visualization, and other utility functions based on Keras and TensorFlow2.

To get started, clone to your local system and install in developer mode::

$pip install -e ./histomics_detect

To run in Docker, mount the folder containing the cloned repository and then pip install
inside the running container.

See /histomics_detect/example/ for a Jupyter notebook demonstrating how to build a FasterRCNN 
model using histomics_detect.
