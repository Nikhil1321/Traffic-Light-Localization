#!/usr/bin/env python3

"""
Created on Tue April 26 2022
@author: Nikhil Raj & Ronak Satpute

This is the Global Varible file.
Description: This file initiates sets up the Global Variables needed for
             execution. These can be modified during runtime if needed.

"""

import numpy as np

# Subscriber Topic
POINT_CLOUD_SUB_TOPIC = '/vehicles/Ego/lidar/pointcloud'
BINARY_MASK_SUB_TOPIC = '/vehicles/Ego/camera/semantic_segmentation/image'
CAMERA_INFO_SUB_TOPIC = '/vehicles/Ego/camera/semantic_segmentation/camera_info'


# Image Dimensions (can be modified during runime)
HEIGHT = 1080
WIDTH = 1920

# Constants for adjusting the threshold for finding intersection
MAX_TOL = 2
REL_TOL = 0.0005

# Default Projection Matrix (will be modified during runtime)
PROJETION_MATRIX = [1662, 0, 960, 0, 0, 1568, 540, 0, 0, 0, 1, 0]

# Reject any contour that is under the threshold area (as it will be too small)
AREA_THRESHOLD = 5.0

# Default value in place where computation does not lead to any viable result
IGNORED_DEFAULT_VALUES = np.ones((9), dtype = np.float32) * 300000

# 3D Spatial Coordinate Computation Node
NODE_NAME = 'traffic_light_position3d'

# Publisher Constants
TOPIC_NAME = 'tl_3dPosition'