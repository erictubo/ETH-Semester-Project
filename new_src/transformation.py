#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from camera import Camera


class Transformation:
    
    """
    Methods for transformations between coordinate frames and pixels
    """

    @staticmethod
    def compile_H_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compiles translation vector (3x1) and rotation matrix (3x3) into homogeneous transformation matrix (4x4),
        without changing frames.
        """
        assert R.shape == (3,3)
        assert t.shape == (3,1)
        H = np.eye(4)
        H[0:3, :]  = np.c_[R, t]
        return H
    
    @staticmethod
    def invert_H_matrix(H: np.ndarray):
        """
        Inverts a homogeneous transformation matrix.
        """
        assert H.shape == (4,4)
        R = H[0:2, 0:2]
        t = H[0:3, 3]
        R_inv = R.transpose
        t_inv = - R_inv @ t
        H_inv = np.eye(4)
        H_inv[0:3, :]  = np.c_[R_inv, t_inv]
        return H_inv
    
    @staticmethod
    def transform_coordinates(H_b_a: np.ndarray, P_a: np.ndarray) -> np.ndarray:
        """
        Transforms coordinates from frame a to frame b, given the homogeneous transformation matrix H_b_a.
        """
        assert H_b_a.shape == (4,4)
        assert P_a.shape == (3,1)
        P_b = ( H_b_a @ np.r_[P_a, [[1]]] )[0:2,:]
        return P_b

    @staticmethod
    def camera_point_to_pixel(Camera: Camera, P_cam: np.ndarray) -> np.ndarray:
        assert Camera.K.shape == (3,3)
        assert P_cam.shape == (3,1)
        pixel = (K @ P_cam)[0:1,:] / (K @ P_cam)[2,:]
        return pixel.round()
    
    @staticmethod
    def point_to_pixel(Camera: Camera, H_cam_a: np.ndarray, P_a: np.ndarray) -> np.ndarray:
        P_cam = Transformation.transform_coordinates(H_cam_a, P_a)
        pixel = Transformation.camera_point_to_pixel(Camera, P_cam)
        return pixel
    