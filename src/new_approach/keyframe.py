#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import yaml

from data import path_to_images, path_to_poses
from camera import Camera
from gps import GPS
from image_features import Image
from annotations import Annotations


class KeyFrame:
    """
    Data Object
    - .id
    - .Camera (data object)
    - .Image (data object)
    - .GPS (data object)
    """

    def __init__(self, id: int, Camera: Camera, distorted_annotations: bool = True):
        self.id = id
        self.Camera = Camera
        self.distorted_annotations = distorted_annotations

        # Keyframe data: image & pose
        self.filename = self.__get_filename__()
        self.distorted_image = self.__get_image__()
        self.image = Camera.undistort_image(self.distorted_image)
        
        self.gps_pose = self.__get_gps_pose__()

        # GPS object, which initialises related sub-objects (see gps.py)
        self.GPS = GPS(self.gps_pose)

        # Image object, which initialises related sub-objects (see image.py)
        self.Image = Image(self.image)

        self.Annotations = Annotations(self.image, self.Camera, self.filename, self.distorted_annotations)


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
        image_path = path_to_images + str(self.filename) + '.jpg'
        image = cv2.imread(image_path)
        return image

    def __get_gps_pose__(self):
        pose_path = path_to_poses + str(self.filename) + '.yaml'
        with open(pose_path, 'r') as stream: gps_pose = yaml.safe_load(stream)
        return gps_pose