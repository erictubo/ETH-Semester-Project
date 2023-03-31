#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal:  returns homogeneous transformation matrices from World to Cam, from GPS to World, from GPS to Cam

Input:
    pose_estimates: 1x6 array containing pose estimates from GPS to Camera
    nr_image:       Number of current frame
    camera_nr:      Number of camera, 0 for right and 1 for left camera

Output:
    H_w_cam:        Homogeneous transformation from world into camera frame
    H_gps_w:        Homogeneous transformation from world into gps frame
    H_gps_cam:      Homogeneous transformation from gps into camera
"""

import yaml
import numpy as np
from scipy.spatial.transform import Rotation


def get_hom_transformation_matrices(pose_estimates, nr_image, camera_nr):
    [x_pos, y_pos, z_pos, ang_x, ang_y, ang_z] = pose_estimates
    
    if camera_nr == 0:
        path_poses = str('/Users/eric/Developer/Cam2GPS/poses/')
    else:
        path_poses = str('/Users/eric/Developer/Cam2GPS/poses/')  

    nr_string = str(nr_image).replace('.0', '')
    if len(nr_string) < 6:
        zeros_add = 6 - len(nr_string)
        pose_path = str(nr_string.zfill(zeros_add+len(nr_string)) + '.yaml')        
    total_path = str(path_poses+pose_path)
    with open(total_path, 'r') as stream:
        pose_data = yaml.safe_load(stream) # p_x, p_y, p_z, euler rotation as quaternion
        
        
        
    ''' rotation of gps in world'''
    rot = Rotation.from_quat([pose_data["q_x"], pose_data["q_y"], pose_data["q_z"], pose_data["q_w"]])
    pos = np.array([[pose_data.get('p_x')],[pose_data.get('p_y')],[pose_data.get('p_z')]])
    euler_angles = rot.as_euler('zyx')
    rot_euler = Rotation.from_euler('zyx', [[euler_angles[0], euler_angles[1], euler_angles[2]]])
    [rot_mat] = rot_euler.as_matrix()

    #rotation from gps to world frame
    t_w_gps = np.array([[pos[0][0]], [pos[1][0]], [pos[2][0]]])
    R_w_gps = rot_mat
    H_w_gps = np.eye(4)
    H_w_gps[0:3, :]  = np.c_[R_w_gps, t_w_gps]
    H_gps_w = np.eye(4)
    H_gps_w[0:3, :]  = np.c_[np.matrix.transpose(R_w_gps), -np.matmul(np.matrix.transpose(R_w_gps), t_w_gps)]
    
    
    
    '''rotation of camera frame to gps'''    
    R_gps_camera = np.array([[0, 0, 1],
                             [-1, 0, 0], 
                             [0, -1, 0]])

    t_gps_camera = np.array([[0],[0],[0]])                                    
    H_gps_cam = np.eye(4)
    H_gps_cam[0:3, :]  = np.c_[R_gps_camera, t_gps_camera]
    
    
    
    '''rotation of camera in own frame  - my estimate'''
    rotx = np.array([[1, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x)], [0, np.sin(ang_x), np.cos(ang_x)]])
    roty = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], [0, 1, 0], [-np.sin(ang_y), 0, np.cos(ang_y)]])
    rotz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], [np.sin(ang_z), np.cos(ang_z), 0], [0, 0, 1]])
    R_cam = np.matmul(rotz, np.matmul(roty, rotx))
    t_cam = np.array([[x_pos], [y_pos], [z_pos]])                              # my estimate, x positive
    H_cam = np.eye(4)
    H_cam[0:3, :]  = np.c_[R_cam, t_cam]    
    
    '''total rotation from world to camera '''
    H_w_cam = np.matmul(np.matmul(H_w_gps, H_gps_cam), H_cam)
    
    
    '''rotation from gps to end camera'''    
    H_gps_cam = np.matmul(H_gps_cam, H_cam)


    return H_w_cam, H_gps_w, H_gps_cam
    