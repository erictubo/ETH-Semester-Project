#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal:  returns rotation around y axis of camera frame in rad

Input:
    img:             Image with lines from hough transform drawn inside
    K:               Instrinsic matrix
    canny_img:       Canny transformation of orginial frame
    vanishing_point: Vanishing point of current frame (1x2 array)
    
Output:
    ang_y:          rotation around positive y-axis in rad
"""

import numpy as np

def rot_y_estimate(img, K, canny_img, vanishing_point):
    ang_y = None
    u_framemiddle = img.shape[1]/2
    fov_u = 2*np.arctan(img.shape[1]/(2*K[0][0]))
    gamma = fov_u / img.shape[1]

    if len(vanishing_point)> 0:
        u_diff = u_framemiddle - vanishing_point[0]
#        ang_y = np.arctan2(x_diff, K[0][0])
        ang_y = u_diff*gamma
        
    return ang_y
    
