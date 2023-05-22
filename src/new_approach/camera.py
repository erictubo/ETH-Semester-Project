#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Camera:

    def __init__(self, focal_length, principal_point, image_size, distortion_coefficients, distortion_model='equidistant', camera_model='pinhole'):

        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.distortion_coefficients = distortion_coefficients
        self.distortion_model = distortion_model
        self.camera_model = camera_model
        
        # Before undistortion
        self.alpha_u = focal_length[0]
        self.alpha_v = focal_length[1]
        self.u0      = principal_point[0]
        self.v0      = principal_point[1]

        self.FoV_u = 2*np.arctan(self.u0 / self.alpha_u)
        self.FoV_v = 2*np.arctan(self.v0 / self.alpha_v)
        self.K_distorted = np.array([[self.alpha_u, 0, self.u0],
                                    [0, self.alpha_v, self.v0],
                                    [0, 0, 1]])
        self.D = np.array(self.distortion_coefficients)
        
        # Undistortion
        if self.distortion_model == 'equidistant':
            self.K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K_distorted, self.D, self.image_size, np.eye(3), P=None, balance=0.0, new_size=self.image_size, fov_scale=1.0)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K_distorted, self.D, np.eye(3), self.K, self.image_size, cv2.CV_16SC2)

        else:
            raise Exception("Distortion model not implemented yet")

    def undistort_image(self, distorted_image):
        image = distorted_image.copy()
        undistorted_image = cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_image
        

    def undistort_points(self, points: list[np.ndarray]):

        print(len(points), points)

        points_array = np.squeeze(np.asarray(points))
        print(points_array.shape)

        points_array = np.array([points_array], dtype=np.float32)
        print(points_array.shape)

        new_points_array = cv2.fisheye.undistortPoints(points_array, K=self.K_distorted, D=self.D, R=np.eye(3), P=self.K)

        new_points_array = np.squeeze(new_points_array)
        print(new_points_array.shape)
        
        # convert to list of np.ndarrays for each row in new_points_array
        new_points: list[np.ndarray] = []
        for i in range(new_points_array.shape[0]):
            new_points.append(new_points_array[i,:])
        print(len(new_points))

        return new_points