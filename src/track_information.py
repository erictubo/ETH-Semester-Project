#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal: returns information on detected railway tracks

Input: 
    canny_img:                  Canny transformation of orginial frame
    img:                        Current frame
    
Output:
    foundTrack:                 Boolean whether found tracks
    line_info:                  List with [difference between middle of frame and middle of tracks, gradient of line1, u-coordinates of line1 at bottom of frame, gradient of line2, u-coordinates of line2 at bottom of frame]
    img_copy:                   Copy of current frame with lines used to estimate x position
    track_width_pix:            Track width in pixel
    u_trackmiddle:              U-coordinate of track middle at bottom of frame
   
"""
 
import cv2
import numpy as np
from valid_rail_detection import detect_validrails as ValRal

def get_track_information(canny_img, img):
    img_copy = np.copy(img)
    u_framemiddle = img.shape[1]/2
    track_width = 1.435                                                         #standard width of train tracks in metres

    line_info = []
    u_trackmiddle = 0
    foundTrack = False
    track_width_pix = 0
    
    isImgVisualization = False                                                  #set to "True" to see lines chosen for vanishing point detection
    
    #detect lines in image
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 100, None, minLineLength=120, maxLineGap=6)

    if lines is not None:
        valid_line_list = ValRal(lines, img)
        
        if round(valid_line_list[0][1]) == canny_img.shape[0]:
            [u_cand_left, v_cand_left, u_cand_left2, v_cand_left2] = valid_line_list[0, :]
            [u_cand_right, v_cand_right, u_cand_right2, v_cand_right2] = valid_line_list[1, :]
            
            grad_left= (v_cand_left - v_cand_left2)/(u_cand_left - u_cand_left2)
            grad_right = (v_cand_right - v_cand_right2)/(u_cand_right - u_cand_right2)
            track_width_pix = abs(u_cand_right - u_cand_left)
            
            u_trackmiddle = (u_cand_right + u_cand_left)/2                      #observed middle point of train tracks at bottom of frame
            k_u1 = track_width / track_width_pix                                #conversion factor from pixel in meters
                                                                               
            if round(abs(u_framemiddle - u_trackmiddle)) < 100:                 #difference between frame middle and track middle
                foundTrack = True
                u_toptrack_middle = (u_cand_left2 + u_cand_right2)/2                             
                diff_heading_middle_track_pix = u_framemiddle - u_trackmiddle
                diff_heading_middle_track = diff_heading_middle_track_pix * k_u1                                      #difference of frame middle to track middle in meters
                v_topframe = (v_cand_left2 + v_cand_right2)/2
                line_info = [diff_heading_middle_track_pix, grad_left, u_cand_left, grad_right, u_cand_right] 

                if isImgVisualization:
                    cv2.line(img_copy, (round(u_cand_left), img_copy.shape[0]), (round(u_cand_left2), round(v_cand_left2)), (0,165,255), 4)
                    cv2.line(img_copy, (round(u_cand_right), img_copy.shape[0]), (round(u_cand_right2), round(v_cand_right2)), (0,165,255), 4)
                    #to only draw the two lines used to calculate width, comment these two lines
                    cv2.line(img_copy, (round(u_framemiddle), img_copy.shape[0]), (round(u_framemiddle), 1900),(255, 0, 0), 20)
                    cv2.line(img_copy, (round(u_trackmiddle), img_copy.shape[0]), (round(u_toptrack_middle), round(v_topframe)),(0, 0, 255), 3)
                    
         
    return foundTrack, line_info, img_copy, track_width_pix, u_trackmiddle