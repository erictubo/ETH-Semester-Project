#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from camera import Camera
from transformation import Transformation


class Evaluation:

    # list/history of error values

    # optimisation

    @staticmethod
    def pixel_distance(pixel_1: np.ndarray, pixel_2: np.ndarray) -> float:
        assert (pixel_1.shape and pixel_2.shape) == (2,1)
        u1, v1, u2, v2 = pixel_1[0], pixel_1[1], pixel_2[0], pixel_2[1]
        distance = ((u2-u1)**2 + (v2-v1)**2)**0.5
        return distance

    @staticmethod
    def reprojection_error(Camera: Camera, H_cam_a, P_a, detected_pixel: np.ndarray):
        """
        Output: distance in pixels between detected and reprojected point
        """
        projected_pixel = Transformation.point_to_pixel(Camera, H_cam_a, P_a)
        return Evaluation.pixel_distance(detected_pixel, projected_pixel)
    