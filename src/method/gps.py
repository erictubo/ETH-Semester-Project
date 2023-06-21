#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External libraries
import numpy as np
from scipy.spatial.transform import Rotation

# Data & methods
from transformation import Transformation
from map_info import MapInfo

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from railway import Railway


class GPS:

    def __init__(self, pose):

        self.is_missing_data = False

        # Extracted from pose: heading, rotation, position, H matrix
        try:
            self.heading = self.__extract_heading__(pose)
            self.position = self.__extract_position__(pose)
            self.quaternions = self.__extract_quaternions__(pose)
            
        except:
            self.is_missing_data = True
            return

        # GPS in world frame = world frame to GPS
        self.t_w_gps = self.position
        self.x_w_gps = float(self.position[0])
        self.y_w_gps = float(self.position[1])
        self.z_w_gps = float(self.position[2])

        self.R_w_gps = Transformation.convert_quaternions(self.quaternions, to_type='rotation matrix')
        self.H_w_gps = Transformation.compile_homogeneous_transformation(self.R_w_gps, self.t_w_gps)
        self.H_gps_w = Transformation.invert_homogeneous_transformation(self.H_w_gps)

        # Extracted map information
        self.elevation = MapInfo.get_elevation(self.x_w_gps, self.y_w_gps)


    @staticmethod
    def __extract_heading__(pose) -> float:
        heading = float(pose.get('heading'))
        assert not np.isnan(heading)
        return heading

    @staticmethod
    def __extract_position__(pose) -> np.ndarray[float]:
        position = np.array([[float(pose.get('p_x'))], [float(pose.get('p_y'))], [float(pose.get('p_z'))]]).squeeze()
        assert position.shape == (3,), position.shape
        # assert that no nan
        assert not np.isnan(position.any())
        return position

    @staticmethod
    def __extract_quaternions__(pose) -> np.ndarray[float]:
        quaternions = np.array([[float(pose.get('q_x'))], [float(pose.get('q_y'))], [float(pose.get('q_z'))], [float(pose.get('q_w'))]]).squeeze()
        # quaternions = np.ndarray([pose["q_x"], pose["q_y"], pose["q_z"], pose["q_w"]])
        assert quaternions.shape == (4,), quaternions.shape
        assert not np.isnan(quaternions.any())
        return quaternions
    
    def __get_local_points_in_tracks__(self, railway: 'Railway', r_ahead: float=None, r_behind: float=None, min_points: int=2):
        local_tracks, local_points_in_tracks = railway.get_local_points_in_tracks(self, r_ahead, r_behind, min_points)

        self.local_tracks = local_tracks
        self.local_points_in_tracks = local_points_in_tracks