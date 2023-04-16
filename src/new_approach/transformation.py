#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation

from camera import Camera


class Transformation:
    
    """
    Methods for transformations of points in different coordinate frames and camera pixels
    """


    """
    Compilation and inversion of homogeneous transformation matrices
    """
    
    @staticmethod
    def compile_H_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compiles translation vector (3x1) and rotation matrix (3x3) into homogeneous transformation matrix (4x4),
        without changing frames.
        """
        if type(R) != np.ndarray:
            R = Rotation.as_matrix(R)
            R = R.squeeze()
        assert R.shape == (3,3), R.shape
        assert t.shape[0] == 3, t.shape
        H = np.eye(4)
        H[0:3, :]  = np.c_[R, t]
        return H

    @staticmethod
    def invert_H_matrix(H: np.ndarray) -> np.ndarray:
        """
        Inverts a homogeneous transformation matrix.
        """
        assert H.shape == (4,4), H.shape
        R = H[0:3, 0:3]
        t = H[0:3, 3]
        R_inv = R.transpose()
        t_inv = -R_inv @ t
        H_inv = np.eye(4)
        H_inv[0:3, :]  = np.c_[R_inv, t_inv]
        return H_inv
    

    """
    Conversions between rotation parametrisations
    """

    @staticmethod
    def convert_euler_angles(euler_angles: list[float], to_type: str = "rotation matrix") -> np.ndarray[float]:
        """
        Converts euler angles to other specified parametrisation: "rotation matrix" (default) or "quaternions"
        """
        assert len(euler_angles) == 3, len(euler_angles)

        if to_type == "rotation matrix":
            euler = Rotation.from_euler('zyx', [[euler_angles[0], euler_angles[1], euler_angles[2]]])
            rotation_matrix = euler.as_matrix().squeeze()
            # ang_x, ang_y, ang_z = euler_angles
            # R_x = np.array([[1, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x)], [0, np.sin(ang_x), np.cos(ang_x)]])
            # R_y = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], [0, 1, 0], [-np.sin(ang_y), 0, np.cos(ang_y)]])
            # R_z = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], [np.sin(ang_z), np.cos(ang_z), 0], [0, 0, 1]])
            # rotation_matrix = R_z @ (R_y @ R_x)

            assert rotation_matrix.shape == (3,3), rotation_matrix.shape
            return rotation_matrix
        
        elif to_type == "quaternions":
            pass
        
    @staticmethod
    def convert_quaternions(quaternions: np.ndarray[float], to_type: str = "rotation matrix") -> np.ndarray[float]:
        """
        Converts quaternions to other specified parametrisation:"rotation matrix" (default) or "euler angles"
        """
        euler_angles = quaternions.as_euler('zyx')
        euler = Rotation.from_euler('zyx', [[euler_angles[0], euler_angles[1], euler_angles[2]]])
        if to_type == "euler angles" or "euler":
            return euler
        elif to_type == "rotation matrix" or "matrix":
            rotation_matrix = euler.as_matrix().squeeze()
            assert rotation_matrix.shape == (3,3), rotation_matrix.shape
            return rotation_matrix
        

    @staticmethod
    def convert_rotation_matrix(rotation_matrix, to_type: str = "quaternions"):
        """
        Converts rotation matrix to other specified parametrisation: "quaternions" (default) or "euler angles"
        """
        pass
    

    """
    Transformations between coordinate frames and pixels
    """
    
    @staticmethod
    def transform_point(H_b_a: np.ndarray, P_a: np.ndarray) -> np.ndarray:
        """
        Transform single point from frame a to frame b,
        given the homogeneous transformation matrix H_b_a (= frame b in frame a).
        """
        assert H_b_a.shape == (4,4), H_b_a.shape
        assert P_a.shape[0] == 3, P_a.shape
        P_b = (H_b_a @ np.r_[P_a, [[1]]])[0:3,:]
        assert P_b.shape[0] == 3, P_b.shape
        return P_b
    
    @staticmethod
    def transform_points(H_b_a: np.ndarray, Ps_a: list[np.ndarray]) -> list[np.ndarray]:
        """
        Transform multiple points from frame a to frame b,
        given the homogeneous transformation matrix H_b_a (= frame b in frame a).
        """
        Ps_b = []
        for P_a in Ps_a:
            P_b = Transformation.transform_point(H_b_a, P_a)
            Ps_b.append(P_b)
        return Ps_b

    @staticmethod
    def project_camera_point_to_pixel(Camera: Camera, P_cam: np.ndarray) -> np.ndarray:
        """
        Project single camera point to pixel,
        given the intrinsic camera parameters (Camera object).
        """
        assert Camera.K.shape == (3,3)
        assert P_cam.shape[0] == 3, P_cam.shape

        pixel_lam = Camera.K @ P_cam
        pixel = np.round(pixel_lam[0:2,:] / pixel_lam[2,:])
        assert pixel.shape[0] == 2, pixel.shape
        return pixel
    
    @staticmethod
    def project_camera_points_to_pixels(Camera: Camera, Ps_cam: list[np.ndarray]) -> list[np.ndarray]:
        """
        Project multiple camera points to pixels,
        given the intrinsic camera parameters (Camera object).
        """
        pixels = []
        for P_cam in Ps_cam:
            pixel = Transformation.project_camera_point_to_pixel(Camera, P_cam)
            pixels.append(pixel)
        return pixels


    @staticmethod
    def project_point_to_pixel(Camera: Camera, H_cam_a: np.ndarray, P_a: np.ndarray) -> np.ndarray:
        """
        Project single point from frame a directly to pixel, given:
        - the homogeneous transformation matrix H_cam_a (= camera frame in frame a)
        - the intrinsic camera parameters (via Camera object).
        """
        P_cam = Transformation.transform_points(H_cam_a, P_a)
        pixel = Transformation.project_camera_point_to_pixel(Camera, P_cam)
        return pixel
    
    @staticmethod
    def project_points_to_pixels(Camera: Camera, H_cam_a: np.ndarray, Ps_a: list[np.ndarray]) -> list[np.ndarray]:
        """
        Project multiple points from frame a directly to pixel, given:
        - the homogeneous transformation matrix H_cam_a (= camera frame in frame a)
        - the intrinsic camera parameters (via Camera object).
        """
        Ps_cam = Transformation.transform_points(H_cam_a, Ps_a)
        pixels = Transformation.project_camera_points_to_pixels(Camera, Ps_cam)
        return pixels
