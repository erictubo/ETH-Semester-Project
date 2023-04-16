#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random

from keyframe import KeyFrame
from camera import Camera
from gps import MapInfo
from transformation import Transformation
from visualisation import Visualisation



"""
Camera Parameters
"""
Camera_1 = Camera(alpha_u= 698.8258566937119*2, alpha_v= 698.6495855393557*2, u0= 492.9705850660823*2, v0= 293.3927615928415*2)
# optional: define different cameras with different parameters


"""
Main Loop
"""

for id in range(1,2):

    keyframe = KeyFrame(id, Camera_1)

    # Camera frame in GPS frame
    R_gps_camf = np.array([[0 ,  0, 1],
                          [-1,  0, 0],
                          [0 , -1, 0]])
    t_gps_camf = np.array([[0],[0],[0]])                                    
    H_gps_camf = Transformation.compile_H_matrix(R_gps_camf, t_gps_camf)

    # GPS in world frame
    H_w_gps = keyframe.GPS.H_w_gps
    # World in GPS frame
    H_gps_w = keyframe.GPS.H_gps_w


    """
    Offset calculations ...
    """

    offset_camf_cam = [0, -1.70, 0]
    angles_camf_cam = [-0.00, -0.01, 0.0]

    # Median estimates from report:
    # Camera 1
    # offset_camf_cam = [1.28, -1.05, 0.04]
    # angles_camf_cam = [-0.31, -0.48, 0.0]
    # Camera 2
    # offset_camf_cam = [0.97, -0.99, 0.48]
    # angles_camf_cam = [0.73, 1.57, 0.0]

    # Camera in camera frame
    t_camf_cam = np.array(offset_camf_cam).transpose()
    R_camf_cam = Transformation.convert_euler_angles(angles_camf_cam, to_type="rotation matrix")
    H_camf_cam = Transformation.compile_H_matrix(R_camf_cam, t_camf_cam)

    # Camera in GPS frame
    H_gps_cam = H_gps_camf @ H_camf_cam
    # GPS in camera frame 
    H_cam_gps = Transformation.invert_H_matrix(H_gps_cam)

    # Camera in world frame
    H_w_cam = H_w_gps @ H_gps_cam
    # World in camera frame
    H_cam_w = Transformation.invert_H_matrix(H_w_cam)


    """
    Reproject railway nodes
    """
    # points_w = keyframe.GPS.railway_nodes_nearby_w
    # points_gps = Transformation.transform_points(H_gps_w, points_w)
    # points_cam = Transformation.transform_points(H_cam_w, points_w)

    tracks_nearby = keyframe.GPS.railway_tracks_nearby
    railway_nodes_in_tracks_nearby = keyframe.GPS.railway_nodes_in_tracks_nearby

    visual = keyframe.image

    for track in tracks_nearby:
        nodes = railway_nodes_in_tracks_nearby[track]
        nodes_w = MapInfo.convert_railway_nodes_to_world_coordinates(nodes)
        nodes_gps = Transformation.transform_points(H_gps_w, nodes_w)
        nodes_cam = Transformation.transform_points(H_cam_w, nodes_w)
        pixels = Transformation.project_camera_points_to_pixels(keyframe.Camera, nodes_cam)

        lines = Visualisation.convert_consecutive_pixels_to_lines(pixels)

        random_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    
        # draw each track in a different colour
        # draw as lines instead of points
        visual = Visualisation.draw_pixels_on_image(visual, pixels, random_color)
        visual = Visualisation.draw_lines_on_image(visual, lines, random_color)

    cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/railway_nodes/" + (3-len(str(id)))*"0"+str(id) + "_railway_tracks.jpg", visual)


    # pixels = Transformation.project_camera_points_to_pixels(keyframe.Camera, points_cam)


    """
    Visualise results
    """
    # Visualisation.show_3D_plot_of_points(points_gps, "Points in GPS frame")

    # image_with_nodes = Visualisation.draw_pixels_on_image(keyframe.image, pixels)
    # cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/railway_nodes/" + (3-len(str(id)))*"0"+str(id) + "_railway_nodes.jpg", image_with_nodes)

print("Finished")
