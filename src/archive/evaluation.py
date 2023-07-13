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
    def reprojection_error(Camera: Camera, H_cam_a: np.ndarray, P_a: np.ndarray, detected_pixel: np.ndarray):
        """
        Output: distance in pixels between detected and reprojected point
        """
        projected_pixel = Transformation.point_to_pixel(Camera, H_cam_a, P_a)
        error = Evaluation.pixel_distance(detected_pixel, projected_pixel)
        return error
    
    @staticmethod
    def reprojection_errors(Camera: Camera, H_cam_a: np.ndarray, Ps_a: list[np.ndarray], detected_pixels: list[np.ndarray]):

        assert len(Ps_a) == len(detected_pixels)
        errors: list[float] = []
        for detected_pixel, P_a in detected_pixels, Ps_a:
            error = Evaluation.reprojection_error(Camera, H_cam_a, P_a, detected_pixel)
            errors.append(error)
        return errors


    # Reprojection error to closest pixel

    # Pass each track through function

    
    @staticmethod
    def average_reprojection_error(Camera: Camera, H_cam_a: np.ndarray, Ps_a: list[np.ndarray], detected_pixels: list[np.ndarray]):
        for i in range(len(detected_pixels)):
            error = Evaluation.reprojection_error(Camera, H_cam_a, Ps_a[i], detected_pixels[i])
        average_error = error/len(detected_pixels)
        return average_error