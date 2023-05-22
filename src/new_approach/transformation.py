#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev, CubicSpline

from camera import Camera
from data import track_width


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
        rotation = Rotation.from_euler('zyx', [[euler_angles[0], euler_angles[1], euler_angles[2]]])

        if to_type == "rotation matrix":
            rotation_matrix = rotation.as_matrix().squeeze()
            assert rotation_matrix.shape == (3,3), rotation_matrix.shape
            return rotation_matrix
        elif to_type == "quaternions":
            quaternions = rotation.as_quat().squeeze()
            assert quaternions.shape == (4,), quaternions.shape
            return quaternions
        else:
            raise ValueError("Unknown conversion type: " + str(to_type))
        
        
    @staticmethod
    def convert_quaternions(quaternions: np.ndarray[float], to_type: str = "rotation matrix") -> np.ndarray[float]:
        """
        Converts quaternions to other specified parametrisation:"rotation matrix" (default) or "euler angles"
        """
        assert quaternions.shape == (4,), quaternions.shape
        rotation = Rotation.from_quat(quaternions)

        if to_type == "euler angles" or "euler":
            euler_angles = rotation.as_euler('zyx')
            assert euler_angles.shape == (3,), euler_angles.shape
            return euler_angles
        elif to_type == "rotation matrix" or "matrix":
            rotation_matrix = rotation.as_matrix().squeeze()
            assert rotation_matrix.shape == (3,3), rotation_matrix.shape
            return rotation_matrix
        else:
            raise ValueError("Unknown conversion type: " + str(to_type))
        

    @staticmethod
    def convert_rotation_matrix(rotation_matrix, to_type: str = "quaternions"):
        """
        Converts rotation matrix to other specified parametrisation: "quaternions" (default) or "euler angles"
        """
        assert rotation_matrix.shape == (3,3), rotation_matrix.shape
        rotation = Rotation.from_matrix(rotation_matrix)

        if to_type == "quaternions":
            quaternions = rotation.as_quat().squeeze()
            assert quaternions.shape == (4,), quaternions.shape
            return quaternions
        elif to_type == "euler angles":
            euler_angles = rotation.as_euler('zyx')
            assert euler_angles.shape == (3,), euler_angles.shape
            return euler_angles
        else:
            raise ValueError("Unknown conversion type: " + str(to_type))
    

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
        pixel = np.round(pixel_lam[0:2,:] / pixel_lam[2,:], 1)
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
        # print(len(pixels), "/", len(Ps_cam), "points in front of camera")
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
    def convert_points_to_coordinate_arrays(points: list[np.ndarray]):
        """ Converts points (list of numpy vectors) into a separate vector for each coordinate, (X,Y) or (X,Y,Z)"""
        
        dim = points[0].shape[0]
        
        X = np.zeros(len(points))
        Y = np.zeros(len(points))
        for i, point in enumerate(points):
            X[i] = point[0]
            Y[i] = point[1]
        if dim == 2:
            return (X, Y)
        elif dim == 3:
            Z = np.zeros(len(points))
            for i, point in enumerate(points):
                Z[i] = point[2]
            return (X, Y, Z)
        else: raise TypeError("Input points not of dimension 2 or 3")


    @staticmethod
    def convert_coordinate_arrays_to_points(x: tuple[np.ndarray]):
        """ Converts vectors per coordinate, (X,Y,Z) or (X,Y), into points (list of numpy vectors)"""
        points: list[np.ndarray] = []
        for i in range(x[0].shape[0]):
            if len(x) == 3:
                point = np.array([[x[0][i]], [x[1][i]], [x[2][i]]])
            elif len(x) == 2:
                point = np.array([[x[0][i]], [x[1][i]]])
            points.append(point)
        return points
    

    @staticmethod
    def interpolate_spline_linspace(points: list[np.ndarray], desired_spacing, smoothing, maximum=False):

        """Spline interpolation for 2D and 3D cases -- matches input dimension."""

        assert(len(points)) > 1, print("Provide at least 2 points for interpolation")

        # covered_distance = sum((points[0]-points[-1])**2)**0.5

        covered_distance = 0
        for i in range(1, len(points)):
            covered_distance += np.linalg.norm(points[i]-points[i-1])

        x = Transformation.convert_points_to_coordinate_arrays(points)

        # Cubic spline for at least 4 points, linear for 2 or 3 points
        if len(points) > 3:
            k = 3
        elif len(points) > 1:
            k = 1

        w = np.ones_like(x[0])
        # w[0], w[-1] = 10, 10

        tck, u = splprep(x=x, w=w, s=smoothing, k=k)

        # If interpolated to fewer points than before, keep original points
        num = 1 + int(covered_distance / desired_spacing)
        if maximum:
            num = max(num, len(u))

        new_u = np.linspace(0, 1, num=num)

        (new_x) = splev(new_u, tck)

        new_points = Transformation.convert_coordinate_arrays_to_points(new_x)
        # print(len(points), "->", len(new_points), "interpolated points")

        return new_points
    

    @staticmethod
    def interpolate_spline_logspace(points: list[np.ndarray], factor, base):

        assert(len(points)) > 1, print("Provide at least 2 points for interpolation")

        # reverse list if last point closer than first point

        if np.linalg.norm(points[0]) > np.linalg.norm(points[-1]):
            points = points[len(points) - 1::-1]

        x = Transformation.convert_points_to_coordinate_arrays(points)

        if len(points) > 3:
            k = 3
        elif len(points) > 1:
            k = 1

        w = np.ones_like(x[0])
        w[0], w[-1] = 10, 10

        tck, u = splprep(x=x, w=w, s=0.1, k=k)

        start = points[0]
        end = points[-1]
        distance = end[2] - start[2]
        average = (end[2] + start[2])/2

        # print(start[2], end[2], distance, average)

        new_u = np.logspace(0, 1, num=int(1000*distance/average+2), base=base)

        new_u = (new_u - base**0)/base**1

        # print("\nNew u:", new_u)

        (new_x) = splev(new_u, tck)
        new_points = Transformation.convert_coordinate_arrays_to_points(new_x)

        return new_points


    @staticmethod
    def separate_track_into_left_right(points: list[np.ndarray]):

        left_points: list[np.ndarray] = []
        right_points: list[np.ndarray] = []

        for i, point in enumerate(points):

            if i == 0:
                gradient = points[i+1] - points[i]
            elif i == len(points)-1:
                gradient = points[i] - points[i-1]
            else:
                gradient = points[i+1] - points[i-1]

            gradient[2] = 0
            gradient = gradient / np.linalg.norm(gradient)

            perpendicular = np.zeros([3,1])
            perpendicular[0] = gradient[1]
            perpendicular[1] = -gradient[0]
            
            left_point = point + perpendicular * track_width/2
            right_point = point - perpendicular * track_width/2

            left_points.append(left_point)
            right_points.append(right_point)


        return left_points, right_points



    

    # @staticmethod
    # def interpolate_3D_spline(points: list[np.ndarray], factor):

    #     X, Y, Z = Transformation.convert_points_to_coordinate_arrays(points)

    #     if len(points) > 3: k = 3
    #     elif len(points) > 1: k = 1

    #     w = np.ones_like(X)

    #     s = 0.1

    #     tck, u = splprep(x=[X, Y, Z], w=w, s=s)

    #     new_u = np.linspace(0, 1, factor * len(u))

    #     new_x = splev(new_u, tck)

    #     new_points = Transformation.convert_coordinate_arrays_to_points(new_x)
    #     return new_points
    

    # @staticmethod
    # def interpolate_2D_spline(points: list[np.ndarray], factor: int):

    #     U, V = Transformation.convert_points_to_coordinate_arrays(points)

    #     if len(points) > 3: k = 3
    #     elif len(points) > 1: k = 1

    #     tck, params = splprep([U, V], s=0.1, k=k)

    #     new_params = np.linspace(0, 1, factor * len(params))

    #     new_U, new_V = splev(new_params, tck)

    #     new_points = Transformation.convert_coordinate_arrays_to_points((new_U, new_V))
    #     return new_points
    
