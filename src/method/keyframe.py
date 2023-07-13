#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import yaml
import random

from data import path_to_images_0, path_to_images_1, path_to_poses

from gps import GPS
from annotations import Annotations
from transformation import Transformation
from visualisation import Visualisation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from railway import Railway
    from camera import Camera

class Frame:
    """
    Initialise basic Frame objects to create processed Railway object.
    - ID
    - filename
    - distorted_image
    - gps_pose
    - GPS
    """

    def __init__(self, id: int, include_elevation=True):
        self.id = id
        self.filename = self.__get_filename__()
        self.gps_pose = self.__get_gps_pose__()

        self.gps = GPS(self.gps_pose, include_elevation)

    def __get_filename__(self, digits = 6) -> str:
        assert isinstance(self.id, int)
        assert len(str(self.id)) <= digits
        zeros = (digits - len(str(self.id))) * "0"
        filename = zeros + str(self.id)
        return filename

    def __get_gps_pose__(self):
        pose_path = path_to_poses + str(self.filename) + '.yaml'
        with open(pose_path, 'r') as stream: gps_pose = yaml.safe_load(stream)
        return gps_pose
    

class Keyframe(Frame):
    """
    Initialise more sophisticated Keyframe objects after having processed a Railway object.
    - ID
    - filename
    - distorted_image
    - gps_pose
    - GPS: with local points in tracks
        - GPS.local_tracks
        - GPS.local_points_in_tracks
    - Camera
    - image
    - Annotations
    """

    def __init__(self, id: int, camera_0: 'Camera', camera_1: 'Camera', railway: 'Railway', distorted_annotations: bool = True):

        super().__init__(id=id, include_elevation=True)

        self.camera_0 = camera_0
        self.camera_1 = camera_1
    
        self.distorted_image_0 = self.__get_image__(self.camera_0)
        self.distorted_image_1 = self.__get_image__(self.camera_1)
    
        self.image_0 = camera_0.undistort_image(self.distorted_image_0)
        self.image_1 = camera_1.undistort_image(self.distorted_image_1)

        self.annotations_0 = Annotations(self.image_0, self.camera_0, self.filename, distorted_annotations)
        self.annotations_1 = Annotations(self.image_1, self.camera_1, self.filename, distorted_annotations)
        
        self.gps.__get_local_points_in_tracks__(railway)
        self.points_gps_array_0, self.points_gps_list_0 = self.__process_local_gps_points__(self.camera_0)
        self.points_gps_array_1, self.points_gps_list_1 = self.__process_local_gps_points__(self.camera_1)
    
    def __get_image__(self, camera: 'Camera'):
        if camera.id == 0:
            path_to_images = path_to_images_0
        elif camera.id == 1:
            path_to_images = path_to_images_1
        image_path = path_to_images + str(self.filename) + '.jpg'
        image = cv2.imread(image_path)
        return image

    
    def __process_local_gps_points__(self, camera: 'Camera', interpolate=True, int_spacing=0.05, int_smoothing=0.1,
                                     filter_by_camera_angle=True, filter_angle=0.004,
                                     separate_left_right=True):
        """
        - Processing of local points
            - interpolate (default: True) with options for spacing and smoothing
            - filter_by_camera_angle (default: True), with option for filter_angle
            - separate_left_right tracks (default: True)
        - Output:
            - single array of points (default "single array") -- simple points used for optimisation
            - list of tracks with points ("separate lists") -- better for distinguishing tracks
        """

        points_gps_array = np.empty((0,3))
        points_gps_list: list[list[np.ndarray]] = []

        for i, track in enumerate(self.gps.local_tracks):

            points_w = self.gps.local_points_in_tracks[track]

            # Transform to GPS frame and interpolate to increase density
            points_gps = Transformation.transform_points(self.gps.H_gps_w, points_w)

            if interpolate:
                points_gps = Transformation.interpolate_spline(points_gps, desired_spacing=int_spacing, smoothing=int_smoothing, maximum=False)

            if filter_by_camera_angle:
                # Transform to camera frame and filter out points that are too close to each other
                points_cam = Transformation.transform_points(camera.H_cam_gps, points_gps)

                previous_point = points_cam[-1]
                for i in range(len(points_cam)-2, 0, -1):
                    point = points_cam[i]
                    if np.arccos(np.dot(point, previous_point) / (np.linalg.norm(point) * np.linalg.norm(previous_point))) < filter_angle:
                        points_cam.pop(i)
                    else:
                        previous_point = points_cam[i]

                # Transform back to GPS frame
                points_gps = Transformation.transform_points(camera.H_gps_cam, points_cam)

            if separate_left_right:
                points_gps_L, points_gps_R = Transformation.separate_track_into_left_right(points_gps)
                points_gps_array_track = Transformation.convert_points_list(points_gps_L + points_gps_R, to_type="array")

                points_gps_list.append(points_gps_L)
                points_gps_list.append(points_gps_R)

            else:
                points_gps_array_track = Transformation.convert_points_list(points_gps, to_type="array")
                points_gps_list.append(points_gps)

            points_gps_array = np.append(points_gps_array, points_gps_array_track, axis=0)

        return points_gps_array, points_gps_list
        
        
    """
    PUBLIC METHODS
    """

    def visualise_reprojected_points(self, camera: 'Camera', visual: np.ndarray=None, color: tuple=(0,0,255)):
        """
        Set color to "random" to get a different color for each track
        """
        if camera.id == 0:
            if visual is None:
                visual = self.image_0.copy()
            points_gps_list = self.points_gps_list_0
        elif camera.id == 1:
            if visual is None:
                visual = self.image_1.copy()
            points_gps_list = self.points_gps_list_1

        for points_gps in points_gps_list:
            if color == "random":
                color_track = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            else:
                color_track = color
            pixels = Transformation.project_points_to_pixels(camera, camera.H_cam_gps, points_gps)
            Visualisation.draw_on_image(visual, pixels, False, color_track)
        return visual
        

    def visualise_original_points(self, camera: 'Camera', visual: np.ndarray=None, color: tuple=(0,255,255)):
        if visual is None:
            if camera.id == 0:
                visual = self.image_0.copy()
            elif camera.id == 1:
                visual = self.image_1.copy()

        for track in self.gps.local_tracks:
            points_w = self.gps.local_points_in_tracks[track]
            points_gps = Transformation.transform_points(self.gps.H_gps_w, points_w)
            pixels = Transformation.project_points_to_pixels(camera, camera.H_cam_gps, points_gps)
            Visualisation.draw_on_image(visual, pixels, False, color)
        return visual
    
    def visualise_reprojected_and_original_points(self, camera: 'Camera', visual: np.ndarray=None, colors: list[tuple]=[(0,0,255), (0,255,255)]):
        visual = self.visualise_reprojected_points(camera, visual, colors[0])
        visual = self.visualise_original_points(camera, visual, colors[1])
        return visual