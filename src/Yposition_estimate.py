#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina
Goal: return relative height of camera to GPS in metres

Input:
    img:                Current frame
    canny_img:          Canny transformation of current frame
    track_width_pix:    Track width in pixels
    K:                  Camera intrinsics
    ang_x:              Rotation angle around x axis in rad
    nr_img:             Current frame number
    path_poses:         Path to stored poses of frames
    H_gps_w:            Homogeneous transformation from GPS frame to world frame

Output:
    y_pos:              Y position estimate of camera
    height_camera:      Height of camera measured from floor   
    height_gps:         Height of GPS from the floor
    elevation:          Elevation of floor at current train position             
"""


import os
import numpy as np
import yaml
import math
import urllib.request
import zipfile


def y_estimate(img, canny_img, track_width_pix, K, ang_x, nr_img, path_poses, H_gps_w):
    y_pos = None
    pitch = -ang_x
    track_width = 1.435     #standard width of train tracks in metres
    focal_length_y = K[1][1]
    path_elevationdata = '/home/nicolina/catkin_ws/src/semester_thesis/bagfiles/'
    
    #compute height of camera with width of detected railway tracks
    r = track_width * focal_length_y/ track_width_pix
    alpha = 2*np.arctan2(img.shape[0], (2* focal_length_y))
    height_camera = r* np.cos(np.pi/2 - pitch - alpha/2)
    
    #get current train position
    nr_string = str(nr_img).replace('.0', '')
    if len(nr_string) < 6: 
        zeros_add = 6 - len(nr_string)
        pose_path = str(nr_string.zfill(zeros_add+len(nr_string)) + '.yaml') 
    total_path = str(path_poses + pose_path)
    with open(total_path, 'r') as stream:
        pose_data = yaml.safe_load(stream) # p_x, p_y, p_z, euler rotation as quaternion
    z_gps = pose_data.get('p_z')
    x_gps =  pose_data.get('p_x')
    y_gps =  pose_data.get('p_y')
    
    #get elevation at current train position
    file_name = str('dgm_33' + str(x_gps)[0:3] + '-' + str(y_gps)[0:4])
    local_file_path = str(path_elevationdata + file_name)
    
    if not os.path.exists(local_file_path):
        zip_url = str('https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/' + file_name + '.zip' )
        
        local_path_zip = str(path_elevationdata + file_name + '.zip')
        
        urllib.request.urlretrieve(zip_url, local_path_zip)
        
        with zipfile.ZipFile(local_path_zip, 'r') as zip_ref:
            zip_ref.extractall(local_file_path)
    
    file = open(str(local_file_path+ '/' + file_name +'.xyz'), 'r')
    diff_position = math.inf
    for line in file:
        split_string = line.split()
        x_curr = split_string[0]
        y_curr = split_string[1]
        curr_difference = np.sqrt((x_gps - float(x_curr))**2 + (y_gps - float(y_curr))**2)
        if curr_difference < diff_position:
            elevation = split_string[2]
            diff_position = curr_difference
    file.close()

    #compute relative height
    y_pos = -(height_camera + float(elevation) - z_gps)
    height_gps = z_gps - float(elevation)
    
    return y_pos, height_camera, height_gps, elevation






















