#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from image import Image


class Features(Image):
    """
    Method class for image feature extraction
    """

    # self.canny_default = self.get_canny_image(grey=True, blur_kernel=(3,3), thresholds=[35,200])
    # self.canny_xz_pos  = self.get_canny_image(grey=True, blur_kernel=(13,13), thresholds=[100,200])
    
    """
    Methods
    - canny
    - hough transform: general
    + specific:
        - vertical hough_lines
        - horizontal hough_lines (rail sleepers)
        - longitudinal hough_lines (railways)
    Features
    - vanishing point
    """

    @staticmethod
    def get_canny_image(image, grey=True, blur_kernel=(3,3), thresholds=[35,200]):
        image_copy = np.copy(image)
        if grey:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)                                
        image_blurred = cv2.GaussianBlur(image_copy, ksize=blur_kernel, sigmaX=0)                                    
        canny_image = cv2.Canny(image_blurred, threshold1=thresholds[1], threshold2=thresholds[2])
        return canny_image

    @staticmethod
    def get_hough_lines(canny_image):
        hough_lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 100, None, minLineLength=120, maxLineGap=6)
        return hough_lines
    

    def detect_valid_rails(self, hough_lines, threshold_difference = 500):
        """
        Returns single left and right extended rail tracks for current frame
        Output:
            valid_rails:    2x4 array | rows: left, right | columns: u1, v1, u2, v2
        """

        valid_rails = np.zeros((2,4))
        
        if len(hough_lines) >= 2:

            valid_lines = []
            extended_lines = []

            # Extract hough_lines that belong to center railway
            for i in range(len(hough_lines)):
                u1, v1, u2, v2 = hough_lines[i, 0, :]
                if v1 > v2:
                    v_smaller = v1
                else:
                    v_smaller = v2
                angle_deg = np.arctan2(abs(v1-v2), abs(u1-u2)) * 180/np.pi
                if (angle_deg) > 55 and (angle_deg < 125) and (abs(u2 - u1) >= 25) and (v_smaller > (self.h-100)):
                    length = np.sqrt((u1-u2)**2+(v1-v2)**2)
                    valid_lines.append([u1, v1, u2, v2, length])

            # Extend hough_lines to fill whole image
            if len(valid_lines) >= 2:
                valid_lines = sorted(valid_lines, key = lambda x: x[4], reverse = True)
                for i in range(len(valid_lines)):
                    [u1, v1, u2, v2, length] = valid_lines[i]                 
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
                    u_bottom_extended = (self.h-intersect)/gradient
                    extended_lines.append([u_bottom_extended, self.h, u_top_extended, 0, length])

            # Select one candidate per rail
            counter = 0
            while counter < len(extended_lines):
                left_u1 = 0
                right_u1 = self.w
                for i in range(len(extended_lines)):
                    [u1, v1, u2, v2, length] = extended_lines[i]
                    if (self.w/2 > u1) and ((self.w/2 - u1) < (self.w/2 - left_u1)) and (u2 > u1):
                        left_u1 = u1
                        left_v1 = v1
                        left_u2 = u2
                        left_v2 = v2
                        index_left = i
                    elif (u1 > self.w/2) and ((u1 - self.w/2) < (right_u1 - self.w/2)) and (u1 > u2):
                        right_u1 = u1
                        right_v1 = v1
                        right_u2 = u2
                        right_v2 = v2 
                        index_right = i
                        
                if (right_u1 - left_u1) < threshold_difference:
                    if (right_u1 - self.w/2) > (self.w/2 - left_u1):
                        del extended_lines[index_left]
                    else:
                        del extended_lines[index_right]
                    counter += 1
                else:
                    counter = len(extended_lines)

            if (left_u1 != 0) and (right_u1 != self.w/2):
                if abs(left_u2 - right_u2) > 150:
                    valid_rails = np.array([left_u1 , left_v1 , left_u2 , left_v2 ],
                                            [right_u1, right_v1, right_u2, right_v2])

        return valid_rails


    def get_track_information(hough_lines, canny_image):

        foundTrack = False

        hough_lines = self.hough_lines
        canny = self.canny_image

        [left_u1, left_v1, left_u2, left_v2]     = self.valid_rails[0, :]
        [right_u1, right_v1, right_u2, right_v2] = self.valid_rails[1, :]




        line_info = []
        u_trackmiddle = 0
        foundTrack = False
        track_width_pix = 0
        
        isImgVisualization = True                                                  #set to "True" to see hough_lines chosen for vanishing point detection
        
        #detect hough_lines in image
        hough_lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 100, None, minLineLength=120, maxLineGap=6)

        if hough_lines is not None:
            valid_line_list = ValRal(hough_lines, img)
            
            if round(valid_line_list[0][1]) == self.h:
                [u_cand_left, v_cand_left, u_cand_left2, v_cand_left2] = valid_line_list[0, :]
                [u_cand_right, v_cand_right, u_cand_right2, v_cand_right2] = valid_line_list[1, :]
                
                grad_left = (v_cand_left - v_cand_left2)/(u_cand_left - u_cand_left2)
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

        return foundTrack, line_info, track_pixel_width, u_trackmiddle


    def get_track_middle():
        pass

    
    def get_vanishing_point():
        pass


    def get_visualisation(self, show_rails = True, show_sleepers = True, show_vanishing_point = True):

        visualisation = np.copy(self.image)

        [left_u1, left_v1, left_u2, left_v2]     = self.valid_rails[0, :]
        [right_u1, right_v1, right_u2, right_v2] = self.valid_rails[1, :]

        if show_rails:
            cv2.line(visualisation, (round(left_u1), visualisation.shape[0]),  (round(left_u2),  round(left_v2)),  (0,165,255), 4)
            cv2.line(visualisation, (round(right_u1), visualisation.shape[0]), (round(right_u2), round(right_v2)), (0,165,255), 4)
            #to only draw the two hough_lines used to calculate width, comment these two hough_lines
            cv2.line(visualisation, (round(self.w/2), visualisation.shape[0]), (round(self.w/2), 1900),(255, 0, 0), 20)
            cv2.line(visualisation, (round(u_trackmiddle), visualisation.shape[0]), (round(u_toptrack_middle), round(v_topframe)),(0, 0, 255), 3)

        if show_sleepers:
            pass

        if show_vanishing_point:
            pass

        return visualisation
