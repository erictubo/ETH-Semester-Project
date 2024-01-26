#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random

from keyframe import KeyFrame
from camera import Camera
from gps import MapInfo
from transformation import Transformation
from visualization import Visualisation



"""
Camera Parameters
"""
Camera_1 = Camera(alpha_u= 698.8258566937119*2, alpha_v= 698.6495855393557*2, u0= 492.9705850660823*2, v0= 293.3927615928415*2)
# optional: define different cameras with different parameters


"""
Main Loop
"""

n_segments = {}
n_nodes = {}

ids = np.arange(6, 1928, 50)


for id in ids:
    id = int(id)
    print("\nID:", id)

    keyframe = KeyFrame(id, Camera_1)

    if keyframe.GPS.is_missing_data:
        print("keyframe missing data >> skipped")
        np.delete(ids, np.where(ids == id))
        continue

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

    visual = keyframe.image

    ax = Visualisation.create_3D_plot("GPS points and splines")

    #colors = [(255,255,255), (0,255,0), (255,0,0), (255,255,0), (0, 255, 255), (0, 122, 255)]

    railway_segments = keyframe.GPS.railway_segment_lists


    for i, segment in enumerate(railway_segments):
        print("Segment #", i, ":", str(len(segment)), "nodes")

        nodes_w = MapInfo.convert_railway_nodes_to_world_coordinates(segment)
        nodes_gps = Transformation.transform_points(H_gps_w, nodes_w)

        # Interpolate spline in 3D
        if len(segment) > 3:
            nodes_gps = Transformation.interpolate_3D_spline(nodes_gps)
            Visualisation.plot_3D_points(ax, nodes_gps)

        nodes_cam = Transformation.transform_points(H_cam_gps, nodes_gps)
        pixels = Transformation.project_camera_points_to_pixels(keyframe.Camera, nodes_cam)

        # Interpolate spline in 2D
        if len(pixels) > 3:
            pixels = Transformation.interpolate_2D_spline(pixels)

        lines_2D = Visualisation.convert_consecutive_pixels_to_2D_lines(pixels)

        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        visual = Visualisation.draw_on_image(visual, pixels, lines_2D, color)
        
    
    # Reproject all railway nodes for comparison with splines
    all_railway_nodes_w = keyframe.GPS.railway_nodes_w
    all_pixels = Transformation.project_points_to_pixels(keyframe.Camera, H_cam_w, all_railway_nodes_w)
    visual = Visualisation.draw_on_image(visual, all_pixels, None)

    cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/railway_nodes/" + (3-len(str(id)))*"0"+str(id) + "_railway_tracks.jpg", visual)

    n_segments[id] = len(railway_segments)
    n_nodes[id] = len(keyframe.GPS.railway_nodes)

    Visualisation.show_3D_plot()



from data import railway_map

# all_railway_nodes = MapInfo.select_local_railway_nodes(railway_map.railway_nodes, keyframe.GPS.x_w_gps, keyframe.GPS.y_w_gps, keyframe.GPS.heading, 500, 500)
# print("Total railway nodes:", len(all_railway_nodes))
# all_railway_nodes_w = MapInfo.convert_railway_nodes_to_world_coordinates(all_railway_nodes)
# all_railway_nodes_gps = Transformation.transform_points(H_gps_w, all_railway_nodes_w)
# Visualisation.plot_3D_points(ax, all_railway_nodes_gps)
# Visualisation.show_3D_plot()


print("Summary:")

for id in ids:
    print("ID", id, "-- local segments:", n_segments[id], "| local nodes:", n_nodes[id])

print("Finished")



"""
New Idea:

Loop through keyframes
- find GPS distances between consecutive keyframes
- create combined map of nodes
- collect nodes into tracks
- interpolate tracks >> regularly spaced nodes
    - interpolate in 2D map-space and add 3D via elevation data

Loop through keyframes again:
- select local nodes & tracks
- interpolate in 2D image space
"""