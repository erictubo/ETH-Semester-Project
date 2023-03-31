#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation

from main import path_to_images, path_to_poses
# ??? Import these here?
from map_info import MapInfo
from transformation import Transformation


class KeyFrame:
    """
    Class per keyframe with its specific properties \\
    Initialised for each keyframe = image/pose combination
    """

    def __init__(self, id: int):
        self.id = id
        self.filename = self.__get_filename__()
        self.image = self.__get_image__()
        self.pose = self.__get_gps_pose__()

        self.heading = self.get_heading()
        self.R_gps_w = self.get_rotation_gps_w()
        self.t_gps_w = self.get_position_gps_w()        
        
        self.x_gps_w = self.t_gps_w[0]
        self.y_gps_w = self.t_gps_w[1]
        self.z_gps_w = self.t_gps_w[2]

        self.H_gps_w = Transformation.compile_H_matrix(self.R_gps_w, self.t_gps_w)
        self.H_w_gps = Transformation.invert_H_matrix(self.H_gps_w)

        self.elevation = MapInfo.get_elevation(self.x_gps_w, self.y_gps_w)

        # Parameters to solve for not keyframe specific ... optimisation accross frames
        # How/where to implement this?


    """
    Data imports for each keyframe
    """

    def __get_filename__(self, digits = 6) -> str:
        assert isinstance(self.id, int)
        assert len(str(self.id)) <= digits
        zeros = (digits - len(str(self.id))) * "0"
        filename = zeros + str(self.id)
        return filename

    def __get_image__(self):
        image_path = str(path_to_images + self.filename(self.keyframe_number) + '.jpg')
        image = cv2.imread(image_path)
        return image

    def __get_gps_pose__(self):
        pose_path = str(path_to_poses + self.filename(self.id) + '.yaml')
        with open(pose_path, 'r') as stream: gps_pose = yaml.safe_load(stream)
        return gps_pose
        

    """
    Extraction of GPS Pose
    """

    def __get_heading__(self) -> float:
        heading = self.pose.get('heading')
        return heading

    def __get_position_gps_w__(self) -> np.ndarray[float]:
        position = np.array([[self.pose.get('p_x')], [self.pose.get('p_y')], [self.pose.get('p_z')]])
        return position

    def __get_rotation_gps_w__(self, type: str = "rotation matrix") -> np.ndarray[float]:
        """
        Extracts rotation parameterisation from gps_pose, depending on specified type:
        "quaternions", "euler angles" or "rotation matrix" (default)
        """
        quaternions = Rotation.from_quat([self.pose["q_x"], self.pose["q_y"], self.pose["q_z"], self.pose["q_w"]])
        if type == "quaternion":
            return quaternions
        euler_angles = quaternions.as_euler('zyx')
        euler = Rotation.from_euler('zyx', [[euler_angles[0], euler_angles[1], euler_angles[2]]])
        if type == "euler" or "euler angles":
            return euler
        matrix = euler.as_matrix()
        return matrix
