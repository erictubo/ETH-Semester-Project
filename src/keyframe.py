#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines Frame and Keyframe classes for representing frames and keyframes in the camera localization pipeline.

Frame contains basic information (ID, GPS data) and is used for dense mapping
Keyframe is a more sophisticated object containing images, associated cameras, GPS, and annotations – used for optimization and evaluation.
"""

import numpy as np
import cv2
import yaml
import random

from data import path_to_images_0, path_to_images_1, path_to_poses

from gps import GPS
from annotation import Annotation
from transformation import Transformation
from visualization import Visualization

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from railway import Railway
    from camera import Camera

class Frame:
    """
    Represents a basic frame with ID, filename, GPS pose, and GPS object.
    Used for creating processed Railway objects.
    """
    def __init__(self, id: int, include_elevation=True):
        """
        Initialize a Frame object.
        Args:
            id: Frame ID (int).
            include_elevation: Whether to include elevation data (default: True).
        """
        self.id = id
        self.filename = self.__get_filename__()
        self.gps_pose = self.__get_gps_pose__()

        self.gps = GPS(self.gps_pose, include_elevation)

    def __get_filename__(self, digits = 6) -> str:
        """
        Generate a zero-padded filename string from the frame ID.
        Args:
            digits: Number of digits for zero-padding (default: 6).
        Returns:
            Zero-padded filename string.
        """
        assert isinstance(self.id, int)
        assert len(str(self.id)) <= digits
        zeros = (digits - len(str(self.id))) * "0"
        filename = zeros + str(self.id)
        return filename

    def __get_gps_pose__(self):
        """
        Load the GPS pose for this frame from a YAML file.
        Returns:
            Dictionary with GPS pose data.
        """
        pose_path = path_to_poses + str(self.filename) + '.yaml'
        with open(pose_path, 'r') as stream: gps_pose = yaml.safe_load(stream)
        return gps_pose
    

class Keyframe(Frame):
    """
    Represents a keyframe with images, cameras, GPS, and annotations.
    Used for optimization and evaluation after processing a Railway object.
    """

    def __init__(self, id: int, camera_0: 'Camera', camera_1: 'Camera', railway: 'Railway', distorted_annotation: bool = True):
        """
        Initialize a Keyframe object.
        Args:
            id: Keyframe ID (int).
            camera_0: Camera object for camera 0.
            camera_1: Camera object for camera 1.
            railway: Railway object for local track information.
            distorted_annotation: Whether to undistort annotation points (default: True).
        """
        super().__init__(id=id, include_elevation=True)
        self.camera_0 = camera_0
        self.camera_1 = camera_1

        # self.cameras = [self.camera_0, self.camera_1]
    
        self.distorted_image_0 = self.__get_image__(self.camera_0)
        self.distorted_image_1 = self.__get_image__(self.camera_1)

        # self.distorted_images = [self.distorted_image_0, self.distorted_image_1]
    
        self.image_0 = camera_0.undistort_image(self.distorted_image_0)
        self.image_1 = camera_1.undistort_image(self.distorted_image_1)

        # self.images = [self.image_0, self.image_1]

        self.annotation_0 = Annotation(self.image_0, self.camera_0, self.filename, distorted_annotation)
        self.annotation_1 = Annotation(self.image_1, self.camera_1, self.filename, distorted_annotation)

        # self.annotations = [self.annotation_0, self.annotation_1]
        
        self.gps.__get_local_points_in_tracks__(railway)
        self.points_gps_array_0, self.points_gps_list_0 = self.__process_local_gps_points__(self.camera_0)
        self.points_gps_array_1, self.points_gps_list_1 = self.__process_local_gps_points__(self.camera_1)

        # self.points_gps_arrays = [self.points_gps_array_0, self.points_gps_array_1]
    
    def __get_image__(self, camera: 'Camera'):
        """
        Load the image for this keyframe and camera.
        Args:
            camera: Camera object.
        Returns:
            Image as a numpy array.
        """
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
        Process local GPS points for this keyframe and camera.
        Args:
            camera: Camera object.
            interpolate: Whether to interpolate points (default: True).
            int_spacing: Interpolation spacing (default: 0.05).
            int_smoothing: Interpolation smoothing (default: 0.1).
            filter_by_camera_angle: Whether to filter by camera angle (default: True).
            filter_angle: Angle threshold for filtering (default: 0.004).
            separate_left_right: Whether to separate left/right tracks (default: True).
        Returns:
            Tuple (points_gps_array, points_gps_list):
                points_gps_array: Array of all points in the GPS frame - simple points for optimization
                points_gps_list: List of lists of points in the GPS frame - better for distinguishing tracks (for visualization)
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

    def visualize_reprojected_points(self, camera: 'Camera', visual: np.ndarray=None, color: tuple=(0,0,255)):
        """
        Visualize reprojected GPS points for this keyframe and camera.
        Args:
            camera: Camera object.
            visual: Optional image to draw on.
            color: Color for the points (default: red). Set to "random" to get a different color for each track.
        Returns:
            Image with reprojected points drawn.
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
            Visualization.draw_on_image(visual, pixels, False, color_track)
        return visual
        

    def visualize_original_points(self, camera: 'Camera', visual: np.ndarray=None, color: tuple=(0,255,255)):
        """
        Visualize original (unprocessed) GPS points for this keyframe and camera.
        Args:
            camera: Camera object.
            visual: Optional image to draw on.
            color: Color for the points (default: yellow).
        Returns:
            Image with original points drawn.
        """
        if visual is None:
            if camera.id == 0:
                visual = self.image_0.copy()
            elif camera.id == 1:
                visual = self.image_1.copy()

        for track in self.gps.local_tracks:
            points_w = self.gps.local_points_in_tracks[track]
            points_gps = Transformation.transform_points(self.gps.H_gps_w, points_w)
            pixels = Transformation.project_points_to_pixels(camera, camera.H_cam_gps, points_gps)
            Visualization.draw_on_image(visual, pixels, False, color)
        return visual
    
    def visualize_reprojected_and_original_points(self, camera: 'Camera', visual: np.ndarray=None, colors: list[tuple]=[(0,0,255), (0,255,255)]):
        """
        Visualize both reprojected and original GPS points for this keyframe and camera.
        
        Args:
            camera: Camera object.
            visual: Optional image to draw on.
            colors: List of two colors for reprojected and original points (default: [blue, yellow]).
            
        Returns:
            Image with both reprojected and original points drawn.
        """
        visual = self.visualize_reprojected_points(camera, visual, colors[0])
        visual = self.visualize_original_points(camera, visual, colors[1])
        return visual