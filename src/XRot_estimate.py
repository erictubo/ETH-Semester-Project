#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal: returns rotation around x axis of camera frame in rad

Input:
    img:             Image with lines from hough transform drawn inside
    K:               Instrinsic matrix
    canny_img:       Canny transformation of orginial frame
    vanishing_point: Vanishing point of current frame (1x2 array)
    
Output:
    ang_x:           Rotation around positive x-axis in rad
"""

import numpy as np

def rot_x_estimate(img, K, canny_img, vanishing_point):
    ang_x = None
    focal_length = K[1][1]
    v = img.shape[0]
    fov_v = 2*np.arctan(v/(2*focal_length))
    gamma = fov_v/img.shape[0]
    
    if len(vanishing_point) > 0:
#        ang_x = -np.arctan2(vanishing_point[1] - y/2, focal_length)
        ang_x = (vanishing_point[1] - v/2)*gamma

    return ang_x
    
