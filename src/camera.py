#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Camera:

    def __init__(self, id: int, focal_length: tuple[int], principal_point: list[float], image_size: list[float],
                 distortion_coefficients: list[float], distortion_model='equidistant', camera_model='pinhole',
                 initial_H_gps_cam=np.array([[0,0,1,0], [-1,0,0,0], [0,-1,0,0], [0,0,0,1]])):

        self.id = id
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.distortion_coefficients = distortion_coefficients
        self.distortion_model = distortion_model
        self.camera_model = camera_model
        
        # Before undistortion
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = principal_point[0]
        self.cy = principal_point[1]

        self.FoV_u = 2*np.arctan(self.cx / self.fx)
        self.FoV_v = 2*np.arctan(self.cy / self.fy)
        self.K_distorted = self.__create_intrinsics_matrix__(self.fx, self.fy, self.cx, self.cy)
        self.D = np.array(self.distortion_coefficients)
        
        # Undistortion
        if self.distortion_model == 'equidistant':
            self.K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K_distorted, self.D, self.image_size, np.eye(3), P=None, balance=0.0, new_size=self.image_size, fov_scale=1.0)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K_distorted, self.D, np.eye(3), self.K, self.image_size, cv2.CV_16SC2)
        
            self.fy, self.fy, self.cx, self.cy = self.__get_intrinsics_from_matrix__(self.K)
        else:
            raise Exception("Distortion model not implemented yet")
                
        self.intrinsics_vector = np.array([self.fx, self.fy, self.cx, self.cy])

        from transformation import Transformation

        self.H_gps_cam = initial_H_gps_cam
        self.H_cam_gps = Transformation.invert_homogeneous_transformation(self.H_gps_cam)
        self.pose_vector = Transformation.compile_pose_vector(self.H_cam_gps)


    @staticmethod
    def __create_intrinsics_matrix__(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        return K


    @staticmethod
    def __get_intrinsics_from_matrix__(K: np.ndarray):
        assert K.shape == (3,3), K.shape
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        return fx, fy, cx, cy


    """
    PUBLIC METHODS
    """

    def update_pose(self, new_pose_vector: np.ndarray):
        assert new_pose_vector.shape == self.pose_vector.shape, (new_pose_vector.shape, self.pose_vector.shape)
        self.pose_vector = new_pose_vector

        from transformation import Transformation
        self.H_cam_gps = Transformation.uncompile_pose_vector(self.pose_vector)
        self.H_gps_cam = Transformation.invert_homogeneous_transformation(self.H_cam_gps)


    def undistort_image(self, distorted_image):
        image = distorted_image.copy()
        undistorted_image = cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_image


    def undistort_points(self, points: list[np.ndarray]) -> list[np.ndarray]:
        points_array = np.squeeze(np.asarray(points))
        points_array = np.array([points_array], dtype=np.float32)

        new_points_array = cv2.fisheye.undistortPoints(points_array, K=self.K_distorted, D=self.D, R=np.eye(3), P=self.K)
        new_points_array = np.squeeze(new_points_array)
        
        # convert to list of np.ndarrays for each row in new_points_array
        new_points: list[np.ndarray] = []
        for i in range(new_points_array.shape[0]):
            new_points.append(new_points_array[i,:])

        assert len(points) == len(new_points)

        return new_points
    

    def project_point(self, point_cam: np.ndarray) -> np.ndarray:
        """
        Project single point in camera frame (3,) to pixel (2,),
        given the intrinsic camera parameters (via Camera object).
        If point is behind the camera, it will be ignored.
        """
        assert self.K.shape == (3,3)
        assert point_cam.shape in [(3,), (3,1)], point_cam.shape
        if point_cam[2] < 0:
            # print("Camera point", str(P_cam.transpose()), "ignored since behind camera")
            return
        pixel_lam = self.K @ point_cam
        pixel = np.round(pixel_lam[0:2] / pixel_lam[2], 1)
        pixel = pixel.squeeze()
        assert pixel.shape == (2,), pixel.shape
        return pixel
    

    def project_points(self, points_cam: list[np.ndarray]) -> list[np.ndarray]:
        """
        Project multiple points in the camera frame (3,) to pixels (2,),
        given the intrinsic camera parameters (via Camera object).
        Any points behind the camera will be ignored.
        """
        pixels = []
        for point_cam in points_cam:
            pixel = self.project_camera_point_to_pixel(point_cam)
            if type(pixel) == np.ndarray:
                pixels.append(pixel)
        # print(len(pixels), "/", len(Ps_cam), "points in front of camera")
        return pixels