#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev, CubicSpline

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
        if P_cam[2] < 0:
            # print("Camera point", str(P_cam.transpose()), "ignored since behind camera")
            return
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
            if type(pixel) == np.ndarray:
                pixels.append(pixel)
        print(len(pixels), "/", len(Ps_cam), "points in front of camera")
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


    @staticmethod
    def convert_points_to_XYZ(points: list[np.ndarray]):
        X = np.zeros(len(points))
        Y = np.zeros(len(points))
        Z = np.zeros(len(points))
        for i, point in enumerate(points):
            assert point.shape[0] == 3
            X[i], Y[i], Z[i] = point[0], point[1], point[2]
        return X, Y, Z
    
    @staticmethod
    def convert_XYZ_to_points(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        points: list[np.ndarray] = []
        for i in range(X.shape[0]):
            point = np.array([[X[i]], [Y[i]], [Z[i]]])
            points.append(point)
        return points


    @staticmethod
    def interpolate_3D_spline(points: list[np.ndarray], factor=3):

        X, Y, Z = Transformation.convert_points_to_XYZ(points)

        W = np.ones_like(X)
        W[0] = 4
        W[-1] = 4

        tck, u = splprep(x=[X, Y, Z], w=W, s=0.1)

        new_u = np.linspace(0, 1, factor * len(u))

        new_XYZ = splev(new_u, tck)
        
        new_X = new_XYZ[0]
        new_Y = new_XYZ[1]
        new_Z = new_XYZ[2]

        print("3D spline:", len(X), "->", len(new_X), "interpolated points")

        new_points = Transformation.convert_XYZ_to_points(new_X, new_Y, new_Z)

        return new_points
    

    @staticmethod
    def convert_pixels_to_UV(pixels: list[np.ndarray]):
        U = np.zeros(len(pixels))
        V = np.zeros(len(pixels))
        for i, pixel in enumerate(pixels):
            assert pixel.shape[0] == 2, pixel.shape
            U[i] = pixel[0]
            V[i] = pixel[1]
        return U, V
    
    @staticmethod
    def convert_UV_to_pixels(U: np.ndarray, V: np.ndarray):
        pixels: list[np.ndarray] = []
        for i in range(U.shape[0]):
            pixel = np.array([[U[i]], [V[i]]])
            pixels.append(pixel)
        return pixels


    @staticmethod
    def interpolate_2D_spline(pixels: list[np.ndarray], factor=3):

        U, V = Transformation.convert_pixels_to_UV(pixels)

        tck, params = splprep([U, V])

        new_params = np.linspace(0, 1, factor * len(params))

        new_U, new_V = splev(new_params, tck)

        print("2D spline:", len(U), "->", len(new_U), "interpolated points")

        new_pixels = Transformation.convert_UV_to_pixels(new_U, new_V)
        return new_pixels