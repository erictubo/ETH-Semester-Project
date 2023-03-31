#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal: find vanishing point in pixels

Input:
    img:             Current image frame
    line_info:       List with [x position estimate in pixels, gradient of line1, x coordinates of line1 at bottom of frame, gradient of line2, x coordinates of line2 at bottom of frame]
    K:               Camera instrinsic matrix 3x3
    canny_img:       Canny transformation of img
    
Output:
    img_copy:       Image with lines leading to vanishing point drawn and vanishing point drawn in
    vanishing_point: Location of vanishing point in pixels
"""
import cv2
import numpy as np

def get_vanishing_point(img, line_info, K, canny_img):
    img_copy = np.copy(img)
    [u_pos_pixel, grad1, u_coord1, grad2, u_coord2] = line_info
    img_height = img_copy.shape[0]
    isImgVisualization = True                      # set to "True" to see lines chosen for vanishing point detection
    threshold = 15
    
    #find intersect of train tracks and determine prior vanishing point
    c1 = img_height - grad1 * u_coord1
    c2 = img_height - grad2 * u_coord2
    u_intersect = (c2 - c1)/(grad1 - grad2)
    v_intersect = grad1*u_intersect+c1
    vanishing_point = [u_intersect, v_intersect]
    
     #find more vanishing lines
    valid_lines = []
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 100, None, minLineLength=230, maxLineGap=6) #200, 50, 5 #250 70
        
    if lines is not None:
        for i in range(len(lines)):
            u1,v1,u2,v2 = lines[i,0,:]
            
            #pick lines within certain angle range
            if ((abs(np.arctan2(v1-v2, u1-u2)) > 10*np.pi/180) and (abs(np.arctan2(v1-v2, u1-u2)) < 80*np.pi/180)) or (abs((np.arctan2(v1-v2, u1-u2)) > 100*np.pi/180) and (abs(np.arctan2(v1-v2, u1-u2)) < 170*np.pi/180)):
                if v1 < v2:
                    v_topframe = v1
                    u_topframe = u1
                    v_bottomframe = v2
                    u_bottomframe = u2
                else:
                    v_topframe = v2
                    u_topframe = u2
                    v_bottomframe = v1
                    u_bottomframe = u1
                    
                #extend lines
                gradient = (v_bottomframe - v_topframe) /(u_bottomframe - u_topframe)
                intersect = v_bottomframe - gradient * u_bottomframe
                u_top_extended = -intersect/gradient
                u_bottom_extended = (img.shape[0]-intersect)/gradient
                valid_lines.append([u_bottom_extended, img.shape[0], u_top_extended, 0, gradient, intersect])
                if isImgVisualization: 
                    cv2.line(img, (round(u_bottom_extended), img_height), (round(u_top_extended), 0), (255,165,0), 2)

    sum_u = u_intersect
    sum_v = v_intersect
    counter_v = 1 #prior from railway track intersection
    
    if len(valid_lines) != 0:
        for i in range(len(valid_lines)):
            #intersection with train tracks
            [u_bottom_extended, v_bottom, u_top_extended, v_top, gradient, intersect] = valid_lines[i]
            if abs(gradient - grad1) > 0 :
                u_intersect_track1 = (intersect - c1)/(grad1 -gradient)
                v_intersect_track1 = gradient * u_intersect_track1 + intersect
                if abs(v_intersect - v_intersect_track1) < threshold:
                    sum_v += v_intersect_track1
                    sum_u += u_intersect_track1
                    counter_v += 1
                
            if abs(gradient - grad2) > 0 :
                u_intersect_track2 = (intersect - c2)/(grad2 - gradient)
                v_intersect_track2 = gradient * u_intersect_track2 + intersect
                if abs(v_intersect - v_intersect_track2) < threshold:
                    sum_v += v_intersect_track2
                    sum_u += u_intersect_track2
                    counter_v += 1
            
    
        center_u = sum_u / counter_v
        center_v = sum_v / counter_v
        vanishing_point = [center_u, center_v]  
        
        if isImgVisualization:
            cv2.circle(img_copy, (round(center_u), round(center_v)), 8, (254, 1, 154) , -1)
        
    return vanishing_point, img_copy
    