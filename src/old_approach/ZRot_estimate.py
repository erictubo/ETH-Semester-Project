#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal:   return angle in z-axis in degrees

Input:  canny_img:          Canny transformation of current frame
        img:                Original frame
        u_trackmiddle:      U-coordinate of track middle at bottom of frame
        track_width_pix:    Width of tracks in pixels at bottom of frame
        vanishing_point:    1x2 array with u-, v-coordinates of vanishing point
        line_info:          List with [difference between middle of frame and middle of tracks,
                                       gradient of line1, u coordinates of line1 at bottom of frame,
                                       gradient of line2, u coordinates of line2 at bottom of frame]
        
Output: z_ang:              rotation angle around z-axis in rad
        img_zrot:           image with lines used for z-angle computation
"""


import numpy as np
import cv2

def rot_z_estimate(canny_img, img, u_trackmiddle, track_width_pix, vanishing_point, line_info):
    isImgVisualization = True                      # set to "True" to see lines chosen for  yaw detection
    z_ang = None
    img_zrot = np.copy(img)
    
    prep_canny = (canny_img*255).astype(np.uint8)
    [diff_heading_middle_track_pix, grad_left, u_cand_left, grad_right, u_cand_right]  = line_info
    c_left = img.shape[0] - grad_left* u_cand_left
    c_right = img.shape[0] - grad_right* u_cand_right
    
    lines = cv2.HoughLinesP(prep_canny, 1, np.pi/180, 80, None, minLineLength=40, maxLineGap=2)
    counter = 1
    ang_counter = 0
    if lines is not None:
        for i in range(len(lines)):
            u1,v1,u2,v2 = lines[i,0,:]
            if  u2 != u1:
                u1_along_track = (v1-c_left)/grad_left
                u2_along_track = (v1-c_right)/grad_right
                if u1 > u1_along_track and u1 < u2_along_track and v1 > (vanishing_point[1]-50) and v2 > (vanishing_point[1]-50):
                    angle = np.arctan((v2-v1)/(u2-u1))*180/np.pi 
                    if angle in range(-15,15):                              #find lines within limited angle range
                        counter += 1
                        ang_counter += angle
                        if isImgVisualization:
                            cv2.line(img_zrot, (u1, v1), (u2, v2), (225, 0, 0), 4)
                                
    if counter > 1:
        z_angle = ang_counter/counter
        z_ang = -z_angle
        
    return z_ang, img_zrot

"""
Summary:
Hough transform of canny image, filter lines to those of rail sleepers
Average gradient of rail sleepers corresponds to the z-rotation estimate

Questions / Errors:
? Line 36: Counter starts at 1 and not 0, is later used for averaging
? Line 40: lines[i,0,:] should return rho & theta values, not Cartesian coordinates
? Line 44: Shouldn't it be: if u1 < u1_along_track and u2 > u2_along_track
"""