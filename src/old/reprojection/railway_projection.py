#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina
"""
import cv2
import numpy as np
from numpy.linalg import inv
from transformation_matrices import get_hom_transformation_matrices as TransMat

def railway_track_projection(img, pose, K, nr_image, end, camera_nr, node_array):
    track_width = 1.435
    [x_pos, y_pos, z_pos, ang_x, ang_y, ang_z] = pose
    H_w_cam, H_gps_w, H_gps_cam = TransMat(pose, nr_image, camera_nr)
    H_cam_w = inv(H_w_cam)
    H_cam_gps = np.matmul(H_cam_w , inv(H_gps_w))
    
    left_track = []
    right_track = []
    for i in range(node_array.shape[1]):
        world_point = node_array[:, i]
        gps_point = np.matmul(H_gps_w, world_point)
        if i> 0: #only plot nodes seen in front of train   
            left_gps_track = np.zeros_like(gps_point)
            left_gps_track[1] = gps_point[1] + track_width/2
            left_gps_track[0] = gps_point[0]
            left_gps_track[2] = gps_point[2]
            left_gps_track[3] = 1
            right_gps_track = np.zeros_like(gps_point)
            right_gps_track[0] = gps_point[0]
            right_gps_track[2] = gps_point[2]
            right_gps_track[3] = 1
            right_gps_track[1] = gps_point[1] - track_width/2
            
            left_cam = np.matmul(H_cam_gps, left_gps_track)
            right_cam = np.matmul(H_cam_gps, right_gps_track)
            
            left_cam_pix_lambda = np.dot(K, left_cam[0:3])
            right_cam_pix_lambda = np.dot(K, right_cam[0:3])
            left_cam_pix = np.array([[left_cam_pix_lambda[0]/left_cam_pix_lambda[2]], [left_cam_pix_lambda[1]/left_cam_pix_lambda[2]]])
            right_cam_pix = np.array([[right_cam_pix_lambda[0]/right_cam_pix_lambda[2]], [right_cam_pix_lambda[1]/right_cam_pix_lambda[2]]])
            
            if round(left_cam_pix[1][0]) > (img.shape[0]/2 + 10) and round(right_cam_pix[1][0]) > (img.shape[0]/2 + 10): #only takes nodes shortly in front of camera
                left_track.append(left_cam_pix)
                right_track.append(right_cam_pix)
    
#            cv2.circle(img, (round(left_cam_pix[0][0]), round(left_cam_pix[1][0])), 9, (255, 0, 0), -1)
#            cv2.circle(img, (round(right_cam_pix[0][0]), round(right_cam_pix[1][0])), 9, (0, 0, 255), -1)
                
    for i, entry in enumerate(left_track):
        if i < len(left_track) -1:
            left_point = left_track[i]
            next_left_point = left_track[i+1]
            right_point = right_track[i]
            next_right_point = right_track[i+1]
            cv2.line(img, (round(left_point[0][0]), round(left_point[1][0])), (round(next_left_point[0][0]), round(next_left_point[1][0])), (0, 0, 255), 9)
            cv2.line(img, (round(right_point[0][0]), round(right_point[1][0])), (round(next_right_point[0][0]), round(next_right_point[1][0])), (0, 0, 255), 9)
#            cv2.circle(img, (round(left_point[0][0]), round(left_point[1][0])), 5, (255, 0, 0), -1)
#            cv2.circle(img, (round(right_point[0][0]), round(right_point[1][0])), 5, (255, 0, 0), -1)
            
    return img