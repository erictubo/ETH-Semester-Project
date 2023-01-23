#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal: Returns single left and right extended rail tracks for current frame

Input: 
    lines:          lines from Hough transformation of current frame
    canny_img:      Canny transformation of current frame

Output:
    valid_rails:    2x4 array, left and right trails defined by start coordinates and end coordinates of line
"""

import numpy as np

def detect_validrails(lines, canny_img):
    valid_rails = np.zeros((2,4))
    threshold_difference = 500
    
    if len(lines) >= 2: #need atleast two lines detected for rails
        valid_lines = []
        extended_lines = []
        u_middle = canny_img.shape[1]/2
        v_middle = canny_img.shape[0]/2

        #extract lines that belong to center railway
        for i in range(len(lines)):
            u1,v1,u2,v2 = lines[i,0,:]
            if v1 > v2:
                v_smaller = v1
            else:
                v_smaller = v2
            if (np.arctan2(abs(v1-v2), abs(u1-u2)) > (55* np.pi/180)) and (np.arctan2(abs(v1-v2), abs(u1-u2)) < (125*np.pi/180)) and abs(u2 -u1) >= 25 and v_smaller > (v_middle-100):
                length = np.sqrt((u1-u2)**2+(v1-v2)**2)
                valid_lines.append([u1, v1, u2, v2, length])
        
        #extend lines to fill whole image
        if len(valid_lines) >= 2:
            valid_lines = sorted(valid_lines, key = lambda x: x[4], reverse = True)
            for i in range(len(valid_lines)):
                [u1,v1,u2,v2, length] = valid_lines[i]                 
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
                gradient = (v_bottomframe - v_topframe) /(u_bottomframe - u_topframe)
                intersect = v_bottomframe - gradient * u_bottomframe
                u_top_extended = -intersect/gradient
                u_bottom_extended = (canny_img.shape[0]-intersect)/gradient
                extended_lines.append([u_bottom_extended, canny_img.shape[0], u_top_extended, 0, length])
            
            #select one candidate line per rail
            counter = 0
            while counter < len(extended_lines):
                u_cand_left = 0
                u_cand_right = canny_img.shape[1]
                for i in range(len(extended_lines)):
                    [u1, v1, u2, v2, length] = extended_lines[i]
                    if np.sign(u_middle - u1) == 1  and (u_middle - u1) < (u_middle - u_cand_left) and np.sign(u2-u1) == 1:
                        u_cand_left = u1
                        v_cand_left = v1
                        u_cand_left2 = u2
                        v_cand_left2 = v2
                        index_left = i
                        
                    elif np.sign(u1 - u_middle) == 1  and (u1 - u_middle) < (u_cand_right - u_middle) and np.sign(u2-u1) == -1:
                        u_cand_right = u1
                        v_cand_right = v1
                        u_cand_right2 = u2
                        v_cand_right2 = v2 
                        index_right = i
                        
                if u_cand_right - u_cand_left < threshold_difference:
                    if (u_cand_right - u_middle) > (u_middle - u_cand_left):
                        del extended_lines[index_left]
                    else:
                        del extended_lines[index_right]
                    counter += 1
                else:
                    counter = len(extended_lines)
                      
            
            if (u_cand_left != 0) and (u_cand_right != canny_img.shape[1]):
                if abs(u_cand_left2 - u_cand_right2) > 150:
                    valid_rails[0, :] = np.array([u_cand_left, v_cand_left, u_cand_left2, v_cand_left2])
                    valid_rails[1, :] = np.array([u_cand_right, v_cand_right, u_cand_right2, v_cand_right2])
            
    return valid_rails