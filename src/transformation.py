#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the Transformation class for geometric operations, coordinate transforms, and spline interpolation.

This module contains static methods for working with homogeneous transformations, rotation representations, 
coordinate frame conversions, and splines. Used throughout the pipeline for geometric computations including
camera projections, coordinate transformations, and track processing.

Classes:
    Transformation: Static utility class for geometric transformations and coordinate operations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev

from camera import Camera
from data import track_width


class Transformation:
    """
    Static utility class for geometric transformations, coordinate conversions, and splines.
    
    This class provides comprehensive tools for:
    - Homogeneous transformation matrix operations
    - Rotation representation conversions (Euler angles, quaternions, rotation matrices)
    - Coordinate frame transformations
    - Camera projection operations
    - Point representation conversions
    - Spline interpolation for track processing
    
    All methods are static and can be called without instantiating the class.
    """

    """
    1. Operations for homogeneous transformation matrices
    """
    
    @staticmethod
    def compile_homogeneous_transformation(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compiles a translation vector and rotation matrix into a homogeneous transformation matrix.
        
        Args:
            R: Rotation matrix of shape (3, 3)
            t: Translation vector of shape (3,) or (3, 1)
            
        Returns:
            Homogeneous transformation matrix of shape (4, 4)
            
        Raises:
            AssertionError: If R is not (3, 3) or t is not 3-dimensional
        """
        # if type(R) != np.ndarray:
        #     R = Rotation.as_matrix(R)
        #     R = R.squeeze()
        assert R.shape == (3,3), R.shape
        assert t.shape[0] == 3, t.shape
        H = np.eye(4)
        H[0:3, :]  = np.c_[R, t]
        return H
    
    @staticmethod
    def separate_homogeneous_transformation(H: np.ndarray):
        """
        Separates a homogeneous transformation matrix into rotation matrix and translation vector.
        
        Args:
            H: Homogeneous transformation matrix of shape (4, 4)
            
        Returns:
            tuple: (R, t) where R is rotation matrix (3, 3) and t is translation vector (3,)
            
        Raises:
            AssertionError: If H is not (4, 4)
        """
        assert H.shape == (4,4), H.shape
        R = H[0:3, 0:3]
        t = np.squeeze(H[0:3, 3])
        return R, t

    @staticmethod
    def invert_homogeneous_transformation(H: np.ndarray) -> np.ndarray:
        """
        Inverts a homogeneous transformation matrix.
        
        Args:
            H: Homogeneous transformation matrix of shape (4, 4)
            
        Returns:
            Inverted homogeneous transformation matrix of shape (4, 4)
            
        Raises:
            AssertionError: If H is not (4, 4)
        """
        assert H.shape == (4,4), H.shape
        R = H[0:3, 0:3]
        t = H[0:3, 3]
        R_inv = R.transpose()
        t_inv = -R_inv @ t
        H_inv = np.eye(4)
        H_inv[0:3, :]  = np.c_[R_inv, t_inv]
        return H_inv
    
    @staticmethod
    def compile_pose_vector(H: np.ndarray):
        """
        Compiles homogeneous transformation matrix into pose vector format.
        
        Args:
            H: Homogeneous transformation matrix of shape (4, 4)
            
        Returns:
            Pose vector of shape (7,) in format [x, y, z, qw, qx, qy, qz]
            
        Raises:
            AssertionError: If H is not (4, 4) or output is not (7,)
        """
        R, t = Transformation.separate_homogeneous_transformation(H)
        q_xyzw = Transformation.convert_rotation_matrix(R)
        # Change quaternion sequence to match Ceres
        q_wxyz = np.append(q_xyzw[3], q_xyzw[0:3])
        assert q_wxyz[0] == q_xyzw[3]
        pose_vector = np.append(t, q_wxyz)
        assert pose_vector.shape == (7,), pose_vector.shape
        return pose_vector
    
    @staticmethod
    def uncompile_pose_vector(pose_vector: np.ndarray):
        """
        Uncompiles pose vector into homogeneous transformation matrix.
        
        Args:
            pose_vector: Pose vector of shape (7,) in format [x, y, z, qw, qx, qy, qz]
            
        Returns:
            Homogeneous transformation matrix of shape (4, 4)
            
        Raises:
            AssertionError: If pose_vector is not (7,)
        """
        assert pose_vector.shape == (7,), pose_vector.shape
        t = pose_vector[0:3]
        q_wxyz = pose_vector[3:7]
        q_xyzw = np.append(q_wxyz[1:4], q_wxyz[0])
        assert q_xyzw[3] == q_wxyz[0]
        R = Transformation.convert_quaternions(q_xyzw)
        H = Transformation.compile_homogeneous_transformation(R, t)
        return H
    
    
    """
    2. Conversions between different rotation parametrisations: euler angles, matrix, quaternions
    """

    @staticmethod
    def convert_euler_angles(euler_angles: list[float], to_type: str = "rotation matrix") -> np.ndarray[float]:
        """
        Converts euler angles to other rotation representations.
        
        Args:
            euler_angles: List of 3 euler angles [yaw, pitch, roll] in radians
            to_type: Target representation - "rotation matrix" (default) or "quaternions" (xyzw)
            
        Returns:
            Rotation matrix of shape (3, 3) or quaternions of shape (4,) in xyzw format
            
        Raises:
            AssertionError: If euler_angles is not length 3
            ValueError: If to_type is not recognized
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
        Converts quaternions to other rotation representations.
        
        Args:
            quaternions: Quaternions of shape (4,) in xyzw format
            to_type: Target representation - "rotation matrix" (default) or "euler angles"
            
        Returns:
            Rotation matrix of shape (3, 3) or euler angles of shape (3,) [yaw, pitch, roll]
            
        Raises:
            AssertionError: If quaternions is not shape (4,)
            ValueError: If to_type is not recognized
        """
        assert quaternions.shape == (4,), quaternions.shape
        rotation = Rotation.from_quat(quaternions)

        if to_type == "rotation matrix":
            rotation_matrix = rotation.as_matrix().squeeze()
            assert rotation_matrix.shape == (3,3), rotation_matrix.shape
            return rotation_matrix
        elif to_type == "euler angles":
            euler_angles = rotation.as_euler('zyx')
            assert euler_angles.shape == (3,), euler_angles.shape
            return euler_angles
        else:
            raise ValueError("Unknown conversion type: " + str(to_type))

    @staticmethod
    def convert_rotation_matrix(rotation_matrix, to_type: str = "quaternion"):
        """
        Converts rotation matrix to other rotation representations.
        
        Args:
            rotation_matrix: Rotation matrix of shape (3, 3)
            to_type: Target representation - "quaternions" (xyzw) (default) or "euler angles"
            
        Returns:
            Quaternions of shape (4,) in xyzw format or euler angles of shape (3,) [yaw, pitch, roll]
            
        Raises:
            AssertionError: If rotation_matrix is not shape (3, 3)
            ValueError: If to_type is not recognized
        """
        assert rotation_matrix.shape == (3,3), rotation_matrix.shape
        rotation = Rotation.from_matrix(rotation_matrix)

        if to_type == "quaternion":
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
    3. Transformations of points between coordinate frames and to/from pixels in image space.
    """
    
    @staticmethod
    def transform_point(H_b_a: np.ndarray, point_a: np.ndarray) -> np.ndarray:
        """
        Transform single point from frame a to frame b using homogeneous transformation.
        
        Args:
            H_b_a: Homogeneous transformation matrix from frame a to frame b, shape (4, 4)
            point_a: Point in frame a, shape (3,) or (3, 1)
            
        Returns:
            Point in frame b, shape (3,)
            
        Raises:
            AssertionError: If H_b_a is not (4, 4) or point_a is not 3-dimensional
        """
        assert H_b_a.shape == (4,4), H_b_a.shape
        assert point_a.shape in [(3,), (3,1)], point_a.shape
        if point_a.shape == (3,): point_a = np.expand_dims(point_a, axis=1)
        point_b = (H_b_a @ np.r_[point_a, [[1]]])[0:3,:]
        point_b = point_b.squeeze()
        assert point_b.shape == (3,), point_b.shape
        return point_b
    
    @staticmethod
    def transform_points(H_b_a: np.ndarray, points_a: list[np.ndarray]) -> list[np.ndarray]:
        """
        Transform multiple points from frame a to frame b using homogeneous transformation.
        
        Args:
            H_b_a: Homogeneous transformation matrix from frame a to frame b, shape (4, 4)
            points_a: List of points in frame a, each of shape (3,)
            
        Returns:
            List of points in frame b, each of shape (3,)
        """
        points_b = []
        for point_a in points_a:
            point_b = Transformation.transform_point(H_b_a, point_a)
            points_b.append(point_b)
        return points_b

    @staticmethod
    def project_camera_point_to_pixel(camera: Camera, point_cam: np.ndarray) -> np.ndarray:
        """
        Project single point in camera frame to pixel coordinates.
        
        Args:
            camera: Camera object containing intrinsic parameters
            point_cam: Point in camera frame, shape (3,) or (3, 1)
            
        Returns:
            Pixel coordinates of shape (2,) or None if point is behind camera
            
        Raises:
            AssertionError: If camera.K is not (3, 3) or point_cam is not 3-dimensional
        """
        assert camera.K.shape == (3,3)
        assert point_cam.shape in [(3,), (3,1)], point_cam.shape
        if point_cam[2] < 0:
            # print("Camera point", str(P_cam.transpose()), "ignored since behind camera")
            return
        pixel_lam = camera.K @ point_cam
        pixel = np.round(pixel_lam[0:2] / pixel_lam[2], 1)
        pixel = pixel.squeeze()
        assert pixel.shape == (2,), pixel.shape
        return pixel
    
    @staticmethod
    def project_camera_points_to_pixels(camera: Camera, points_cam: list[np.ndarray]) -> list[np.ndarray]:
        """
        Project multiple points in camera frame to pixel coordinates.
        
        Args:
            camera: Camera object containing intrinsic parameters
            points_cam: List of points in camera frame, each of shape (3,)
            
        Returns:
            List of pixel coordinates, each of shape (2,). Points behind camera are excluded.
        """
        pixels = []
        for point_cam in points_cam:
            pixel = Transformation.project_camera_point_to_pixel(camera, point_cam)
            if type(pixel) == np.ndarray:
                pixels.append(pixel)
        # print(len(pixels), "/", len(Ps_cam), "points in front of camera")
        return pixels

    @staticmethod
    def project_point_to_pixel(camera: Camera, H_cam_a: np.ndarray, point_a: np.ndarray) -> np.ndarray:
        """
        Project single point from frame a directly to camera pixel coordinates.
        
        Args:
            camera: Camera object containing intrinsic parameters
            H_cam_a: Homogeneous transformation matrix from frame a to camera frame, shape (4, 4)
            point_a: Point in frame a, shape (3,) or (3, 1)
            
        Returns:
            Pixel coordinates of shape (2,) or None if point is behind camera
        """
        point_cam = Transformation.transform_points(H_cam_a, point_a)
        return Transformation.project_camera_point_to_pixel(camera, point_cam)
    
    @staticmethod
    def project_points_to_pixels(camera: Camera, H_cam_a: np.ndarray, points_a: list[np.ndarray]) -> list[np.ndarray]:
        """
        Project multiple points from frame a directly to camera pixel coordinates.
        
        Args:
            camera: Camera object containing intrinsic parameters
            H_cam_a: Homogeneous transformation matrix from frame a to camera frame, shape (4, 4)
            points_a: List of points in frame a, each of shape (3,)
            
        Returns:
            List of pixel coordinates, each of shape (2,). Points behind camera are excluded.
        """
        points_cam = Transformation.transform_points(H_cam_a, points_a)
        return Transformation.project_camera_points_to_pixels(camera, points_cam)
    

    """
    4. Representation of multiple points:
       - "list" (list of point vectors)
       - "components" (tuple of component vectors for all points)
       - "array" (2D array of all points (rows) with all components (columns))
    """

    @staticmethod
    def convert_points_list(points: list[np.ndarray], to_type="components") -> tuple[np.ndarray]:
        """
        Converts a list of points to a different representation.

        Args:
            points: List of points, each of shape (2,) or (3,)
            to_type: Target representation - 'components' (default) or 'array'

        Returns:
            If to_type is 'components': Tuple of arrays (X, Y, Z) or (X, Y)
            If to_type is 'array': 2D array of shape (n_points, n_dimensions)

        Raises:
            AssertionError: If points have inconsistent dimensions
            ValueError: If to_type is not recognized
        """
        points_array = np.asarray(points)
        if len(points_array.shape) > 2:
            points_array = points_array.squeeze()
        assert points_array.shape == (len(points), points[0].shape[0]), \
            print("Array shape", points_array.shape, "with input list length", len(points), "and first point dimension", points[0].shape)
        if to_type == "array":
            return points_array
        elif to_type == "components":
            return Transformation.convert_points_array(points_array, to_type)
        else:
            raise ValueError("Unknown to_type", to_type)

    @staticmethod
    def convert_points_components(x: tuple[np.ndarray], to_type="list") -> list[np.ndarray]:
        """
        Converts a tuple of component arrays to a list of points.

        Args:
            x: Tuple of arrays (X, Y, Z) or (X, Y) for 2D points
            to_type: Target representation - 'list' (default) or 'array'

        Returns:
            If to_type is 'list': List of points, each of shape (2,) or (3,)
            If to_type is 'array': 2D array of shape (n_points, n_dimensions)

        Raises:
            AssertionError: If component arrays have inconsistent lengths
            ValueError: If to_type is not recognized
        """
        points_array = np.asarray(x).transpose()
        assert points_array.shape == (x[0].shape[0], len(x)), \
            print("Array shape", points_array.shape, "with first component shape", x[0].shape, "and dimensions", len(x))
        if to_type == "array":
            return points_array
        elif to_type == "list":
            return Transformation.convert_points_array(points_array, to_type)
        else:
            raise ValueError("Unknown to_type", to_type)

    @staticmethod
    def convert_points_array(points_array: np.ndarray, to_type: str):
        """
        Converts a 2D array of points to a different representation.

        Args:
            points_array: 2D array of shape (n_points, n_dimensions) where n_dimensions is 2 or 3
            to_type: Target representation - 'list' or 'components'

        Returns:
            If to_type is 'list': List of points, each of shape (2,) or (3,)
            If to_type is 'components': Tuple of arrays (X, Y, Z) or (X, Y)

        Raises:
            AssertionError: If points_array is not 2D or has invalid dimensions
            ValueError: If to_type is not recognized
        """
        n = points_array.shape[0]
        dim = points_array.shape[1]
        assert dim in [2,3], points_array.shape
        if to_type == "list":
            points_list = np.vsplit(points_array, n)
            for i in range(n):
                points_list[i] = np.squeeze(points_list[i])
            return points_list
        elif to_type == "components":
            points_components = np.hsplit(points_array, dim)
            for i in range(dim):
                points_components[i] = points_components[i].squeeze()
            return tuple(points_components)
        else:
            raise ValueError("Unknown to_type", to_type)


    """
    5. Interpolation
    """

    @staticmethod
    def interpolate_spline(points: list[np.ndarray], desired_spacing, smoothing, maximum=False):
        """
        Interpolates a spline through a list of points with specified spacing and smoothing.

        Args:
            points: List of points, each of shape (2,) or (3,)
            desired_spacing: Desired spacing between interpolated points in meters
            smoothing: Smoothing factor for the spline (higher = smoother)
            maximum: If True, returns the maximum arc length instead of interpolated points

        Returns:
            If maximum=False: List of interpolated points, each of shape (2,) or (3,)
            If maximum=True: Maximum arc length of the spline

        Raises:
            AssertionError: If fewer than 2 points are provided
        """
        assert(len(points)) > 1, print("Provide at least 2 points for interpolation")

        # reverse list if last point closer than first point
        if np.linalg.norm(points[0]) > np.linalg.norm(points[-1]):
            points = points[len(points) - 1::-1]

        covered_distance = 0
        for i in range(1, len(points)):
            covered_distance += np.linalg.norm(points[i]-points[i-1])

        x = Transformation.convert_points_list(points, to_type="components")

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

        new_points = Transformation.convert_points_components(new_x, to_type="list")
        # print(len(points), "->", len(new_points), "interpolated points")

        return new_points


    @staticmethod
    def separate_track_into_left_right(points: list[np.ndarray]):
        """
        Separates a track centerline into left and right rails based on track width.

        Args:
            points: List of centerline points, each of shape (3,)

        Returns:
            tuple: (left_points, right_points) where each is a list of points of shape (3,)

        Raises:
            AssertionError: If any point is not 3-dimensional
        """
        points_L: list[np.ndarray] = []
        points_R: list[np.ndarray] = []

        for i, point in enumerate(points):

            assert point.shape in [(3,), (3,1)], point.shape

            if i == 0:
                gradient = points[i+1] - points[i]
            elif i == len(points)-1:
                gradient = points[i] - points[i-1]
            else:
                gradient = points[i+1] - points[i-1]

            gradient[2] = 0
            gradient = gradient / np.linalg.norm(gradient)

            perpendicular = np.zeros((3,))
            perpendicular[0] = gradient[1]
            perpendicular[1] = -gradient[0]
            
            point_L = point + perpendicular * track_width/2
            point_R = point - perpendicular * track_width/2

            assert point_L.shape in [(3,), (3,1)], point_L.shape
            assert point_R.shape in [(3,), (3,1)], point_R.shape

            points_L.append(point_L)
            points_R.append(point_R)

        return points_L, points_R
    

    # @staticmethod
    # def interpolate_spline_logspace(points: list[np.ndarray], factor, base):

    #     assert(len(points)) > 1, print("Provide at least 2 points for interpolation")

    #     # reverse list if last point closer than first point
    #     if np.linalg.norm(points[0]) > np.linalg.norm(points[-1]):
    #         points = points[len(points) - 1::-1]

    #     x = Transformation.convert_points_list(points)

    #     if len(points) > 3:
    #         k = 3
    #     elif len(points) > 1:
    #         k = 1

    #     w = np.ones_like(x[0])
    #     w[0], w[-1] = 10, 10

    #     tck, u = splprep(x=x, w=w, s=0.1, k=k)

    #     start = points[0]
    #     end = points[-1]
    #     distance = end[2] - start[2]
    #     average = (end[2] + start[2])/2

    #     # print(start[2], end[2], distance, average)

    #     new_u = np.logspace(0, 1, num=int(1000*distance/average+2), base=base)

    #     new_u = (new_u - base**0)/base**1

    #     # print("\nNew u:", new_u)

    #     (new_x) = splev(new_u, tck)
    #     new_points = Transformation.convert_points_components(new_x)

    #     return new_points


    # @staticmethod
    # def interpolate_3D_spline(points: list[np.ndarray], factor):

    #     X, Y, Z = Transformation.convert_points_list(points)

    #     if len(points) > 3: k = 3
    #     elif len(points) > 1: k = 1

    #     w = np.ones_like(X)

    #     s = 0.1

    #     tck, u = splprep(x=[X, Y, Z], w=w, s=s)

    #     new_u = np.linspace(0, 1, factor * len(u))

    #     new_x = splev(new_u, tck)

    #     new_points = Transformation.convert_points_components(new_x)
    #     return new_points
    

    # @staticmethod
    # def interpolate_2D_spline(points: list[np.ndarray], factor: int):

    #     U, V = Transformation.convert_points_list(points)

    #     if len(points) > 3: k = 3
    #     elif len(points) > 1: k = 1

    #     tck, params = splprep([U, V], s=0.1, k=k)

    #     new_params = np.linspace(0, 1, factor * len(params))

    #     new_U, new_V = splev(new_params, tck)

    #     new_points = Transformation.convert_points_components((new_U, new_V))
    #     return new_points
    
