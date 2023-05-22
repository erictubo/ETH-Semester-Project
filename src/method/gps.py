#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External libraries
import numpy as np
from scipy.spatial.transform import Rotation

# Data & methods
from transformation import Transformation
from map_info import MapInfo


class GPS:

    """
    Data Object\\
    Initialised automatically by KeyFrame object\\
    Use keyframe.GPS to access data
    """

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
        self.H_w_gps = Transformation.compile_H_matrix(self.R_w_gps, self.t_w_gps)
        self.H_gps_w = Transformation.invert_H_matrix(self.H_w_gps)

        # Extracted map information
        self.elevation = MapInfo.get_elevation(self.x_w_gps, self.y_w_gps)

        # self.railway_nodes = MapInfo.select_local_railway_nodes(railway_map.railway_nodes, self.x_w_gps, self.y_w_gps, self.heading, r_ahead, r_behind)
        # self.railway_nodes_w = MapInfo.convert_railway_nodes_to_world_coordinates(self.railway_nodes)

        # self.railway_tracks, self.railway_nodes_in_tracks = MapInfo.get_railway_nodes_by_tracks(self.x_w_gps, self.y_w_gps, self.heading, r_ahead, r_behind)

        # self.railway_segment_lists = MapInfo.divide_into_continuous_segments(self.railway_tracks, self.railway_nodes_in_tracks)

    @staticmethod
    def __extract_heading__(pose) -> float:
        heading = pose.get('heading')
        return heading

    @staticmethod
    def __extract_position__(pose) -> np.ndarray[float]:
        position = np.array([[float(pose.get('p_x'))], [float(pose.get('p_y'))], [float(pose.get('p_z'))]])
        return position

    @staticmethod
    def __extract_quaternions__(pose) -> np.ndarray[float]:
        quaternions = Rotation.from_quat([pose["q_x"], pose["q_y"], pose["q_z"], pose["q_w"]])
        return quaternions