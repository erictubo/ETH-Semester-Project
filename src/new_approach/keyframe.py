#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import yaml


from data import path_to_images, path_to_poses
from camera import Camera
from image import Image
from gps import GPS


class KeyFrame:
    
    """
    Class per keyframe with its specific properties \\
    Initialised for each keyframe = image/pose combination
    """

    def __init__(self, id: int, Camera: Camera):
        self.id = id
        self.Camera = Camera

        # Keyframe data: image & pose
        self.filename = self.__get_filename__()
        self.image = self.__get_image__()
        self.gps_pose = self.__get_gps_pose__()

        # Pose object, which initialises related sub-objects (see pose.py)
        self.GPS = GPS(self.gps_pose)

        # Image object, which initialises related sub-objects (see image.py)
        self.Image = Image(self.image)

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

    
