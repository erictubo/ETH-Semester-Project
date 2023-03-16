#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import yaml
import urllib.request
import zipfile
import typing
from scipy.spatial.transform import Rotation

"""
Global parameters:
"""

# Intrinsic camera parameters
alpha_u = 698.8258566937119*2
alpha_v = 698.6495855393557*2
u0 =      492.9705850660823*2
v0 =      293.3927615928415*2

K = np.array([[alpha_u, 0      , u0],
              [0      , alpha_v, v0], 
              [0      , 0      , 1 ]])

FoV_u = 2*np.arctan(2*u0 / 2*alpha_u)
FoV_v = 2*np.arctan(2*v0 / 2*alpha_v)

# Known constants
track_width = 1.435
pole_height = 5.3


# To-Do: change paths

path_to_data = '/home/nicolina/catkin_ws/src/semester_thesis/bagfiles/'

# Elevation data
path_to_elevation_data = path_to_data

# Pole locations
path_to_pole_location_data = str(path_to_data + 'full_track.csv')
pole_location_dataframe = pd.DataFrame(pd.read_csv(path_to_pole_location_data), columns=['x', 'y'])


def path_to_images(camera_nr: int):
    return str(path_to_data + 'bas_usb' + str(camera_nr) + '/images/')

def path_to_poses(camera_nr: int):
    return str(path_to_data + 'bas_usb' + str(camera_nr) + '/poses/')


# poses <--> H_gps_w relation ???

# Map information: tracks, etc. ???
# What exactly does this include



"""
Global helper functions for general purposes
- data retrieval
- data processing
- ...

(anything to be used accross files or that is too general to fit into another file)

"""


def get_gps_pose(img_nr: int, camera_nr: int):
    """
    Retrieves GPS pose corresponding to the specified image and camera.
    Output: gps_pose
    """

    nr_string = str(img_nr).replace('.0', '')
    if len(nr_string) < 6: 
        zeros_add = 6 - len(nr_string)
        pose_path = str(nr_string.zfill(zeros_add+len(nr_string)) + '.yaml')
    total_path = str(path_to_poses(camera_nr) + pose_path)
    with open(total_path, 'r') as stream:
        gps_pose = yaml.safe_load(stream)

    return gps_pose


def get_gps_position(gps_pose):
    """
    Extracts position from GPS pose.
    Output: position [x, y, z]
    """
    position = np.array([[gps_pose.get('p_x')],[gps_pose.get('p_y')],[gps_pose.get('p_z')]])


def get_gps_rotation(gps_pose, type: str = "rotation matrix"):
    """
    Extracts rotation parameterisation from gps_pose, depending on specified type.
    Output: quaternion, euler angles or rotation matrix (default)
    """
    quaternion = Rotation.from_quat([gps_pose["q_x"], gps_pose["q_y"], gps_pose["q_z"], gps_pose["q_w"]])
    if type == "quaternion":
        return quaternion
    euler_angles = quaternion.as_euler('zyx')
    euler = Rotation.from_euler('zyx', [[euler_angles[0], euler_angles[1], euler_angles[2]]])
    if type == "euler" or "euler angles":
        return euler
    matrix = euler.as_matrix()
    return matrix


def get_transformation_matrix(gps_pose):
    """
    Converts GPS pose to homogeneous transformation matrix.
    Output: H_gps_w
    """

    position = get_gps_position(gps_pose)
    rotation_matrix = get_gps_rotation(gps_pose, "matrix")

    t_w_gps = np.array([[position[0][0]], [position[1][0]], [position[2][0]]])
    R_w_gps = rotation_matrix
    H_w_gps = np.eye(4)
    H_w_gps[0:3, :]  = np.c_[R_w_gps, t_w_gps]
    H_gps_w = np.eye(4)
    H_gps_w[0:3, :]  = np.c_[np.matrix.transpose(R_w_gps), -np.matmul(np.matrix.transpose(R_w_gps), t_w_gps)]
    
    return H_gps_w




def get_elevation_data(x_gps: float, y_gps: float):
    """
    Retrieves elevation of specified GPS position [x, y].
    Output: elevation
    """

    file_name = str('dgm_33' + str(x_gps)[0:3] + '-' + str(y_gps)[0:4])
    local_file_path = str(path_to_elevation_data + file_name)
    
    if not os.path.exists(local_file_path):
        zip_url = str('https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/' + file_name + '.zip' )
        local_path_zip = str(path_to_elevation_data + file_name + '.zip')
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
            elevation = float(split_string[2])
            diff_position = curr_difference
    file.close()

    return elevation




def find_closest_pole_gps_position(elevation: float, H_gps_w: np.ndarray[float]):
    """
    Determines closest pole to the specified GPS position.
    Output: pole_worldview_gps
    """

    df = pole_location_dataframe
    for index, pole in df.iterrows():
        test_pole_gps_world_view = [df.x[index], df.y[index], elevation, 1]
        test_pole_gps = np.matmul(H_gps_w, test_pole_gps_world_view)
        if smallest_diff > test_pole_gps[0] > 0:
            smallest_diff = test_pole_gps[0]
            smallest_index = index

    x_pole_world = df.x[smallest_index]
    y_pole_world = df.y[smallest_index]
    pole_elevation = get_elevation_data(x_pole_world, y_pole_world)

    pole_worldview_world = [x_pole_world, x_pole_world, pole_elevation, 1]
    pole_worldview_gps = np.matmul(H_gps_w, pole_worldview_world)

    return pole_worldview_gps