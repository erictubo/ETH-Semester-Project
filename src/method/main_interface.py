#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External Libraries
import os
import sys
import numpy as np
import cv2
import random
import pickle

# Methods & Data
from data import railway_object_file
from camera import Camera
from keyframe import KeyFrame, create_keyframes

from railway import Railway
from transformation import Transformation
from visualisation import Visualisation

src_path = os.path.abspath('../LROD_extrinsics_estimation/src')
sys.path.append(src_path)
from cpp.optimization import optimize_camera_pose


"""
Camera pose: initial guess
"""

# Camera frame (camf) = position of GPS frame with rotation to approximate camera frame
# Camera frame in GPS frame
R_gps_camf = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
t_gps_camf = np.array([[0],[0],[0]])                                    
H_gps_camf = Transformation.compile_homogeneous_transformation(R_gps_camf, t_gps_camf)

# offset_camf_cam = [0, -1.70, 0]
# angles_camf_cam = [-0.00, -0.01, 0.0]

offset_camf_cam = [0., -2., -5.]
angles_camf_cam = [0., 0., 0.]

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
H_camf_cam = Transformation.compile_homogeneous_transformation(R_camf_cam, t_camf_cam)

initial_H_gps_cam = H_gps_camf @ H_camf_cam


"""
Camera parameters:
- specific intrinsics, distortion coefficients, initial guess of pose
- camera poses will automatically be optimised separately for each camera,
  assuming the keyframes are initiallised with the correct corresponding cameras
"""

camera_0 = Camera(image_size = (1920, 1200),
                  focal_length = [2*698.8258566937119, 2*698.6495855393557], 
                  principal_point = [2*492.9705850660823, 2*293.3927615928415], 
                  distortion_coefficients = [0.15677829213401465, 0.1193983032005652, -0.011287926707786692, 0.23013426724923478],
                  initial_H_gps_cam=initial_H_gps_cam)

# camera_1 = Camera(image_size = (1920, 1200),
#                   focal_length = [2*698.5551622016458, 2*698.6714179616163],
#                   principal_point = [2*489.7761220331873, 2*297.9103051604121],
#                   distortion_coefficients = [0.16213733886648546, 0.11615082942281711, -0.05779296718987261, 0.324051803251489],
#                   initial_H_gps_cam=initial_H_gps_cam_1
#                   )


"""
Railway pre-processing around selected keyframes
- this will import relevant railway nodes, interpolate gaps between railway nodes,
  add elevation data, create compatible tracks, etc.
- the railway object will be saved to file for future use
- trade-off: requires initial pre-processing time, but enables much faster reprojection loop
"""

ids = np.arange(0, 1928, 25)

max_gap = 20
r_ahead = 100
r_behind = 50

if input("Create a new railway object? [y/n]: ") == 'y':
    keyframes = create_keyframes(ids, camera_0)

    railway = Railway(keyframes, max_gap=max_gap, r_ahead=r_ahead, r_behind=r_behind)
    with open(railway_object_file, 'wb') as file:
        pickle.dump(railway, file)
        print("Railway object sucessfully saved to file.")
else:
    print("Loading railway object from file")
    with open(railway_object_file, 'rb') as file:
        railway = pickle.load(file)


"""
Visualisation of pre-processed railway in 2D and 3D
"""

if input("Plot railway? [y/n]: ") == 'y':
    keyframes = create_keyframes(ids, camera_0)

    for track in railway.tracks:
        for point in railway.points_in_tracks_2D[track]:
            Visualisation.plot_XY(point[0], point[1], 'orange')
    for node in railway.nodes:
        Visualisation.plot_XY(node.x, node.y, color='red')
    for keyframe in keyframes:
        point = keyframe.GPS.t_w_gps
        Visualisation.plot_XY(point[0], point[1], 'black')
        #plt.Circle((point[0], point[1]), r_ahead, 'red')
    Visualisation.show_plot()

    ax = Visualisation.create_3D_plot("All points in tracks")
    for track in railway.tracks:
        color = 'blue'
        if track.is_bridge:
            color = 'cyan'
        points = railway.points_in_tracks_3D[track]
        Visualisation.plot_3D_points(ax, points, color)
    for keyframe in keyframes:
        point = keyframe.GPS.t_w_gps
        Visualisation.plot_3D_points(ax, [point], 'red')
    Visualisation.show_plot()


"""
Reprojection loop for selected keyframes
- subset of keyframes used to generate pre-processed railway map
- if the keyframes are not located within the pre-processed railway map, fewer or no nodes will be found
"""

ids = [325, 450, 600, 775, 1025, 1225, 1650, 1800, 1900]
keyframes = create_keyframes(ids, camera_0)

for i, keyframe in enumerate(keyframes):

    print("Keyframe", i, "/", len(keyframes))

    H_w_cam = keyframe.GPS.H_w_gps @ keyframe.Camera.H_gps_cam
    H_cam_w = Transformation.invert_homogeneous_transformation(H_w_cam)

    reprojected_visual = keyframe.image.copy()

    points_gps_array = np.empty((0,3))
    
    local_tracks, local_points_in_tracks = railway.get_local_points_in_tracks(keyframe, r_ahead, r_behind, min_points=2)
    for track in local_tracks:

        # transform local points from world to gps frame (keyframe specific)
        points_w = local_points_in_tracks[track]

        points_gps = Transformation.transform_points(keyframe.GPS.H_gps_w, points_w)

        # Interpolate 3D points to increase density
        points_gps = Transformation.interpolate_spline(points_gps, desired_spacing=0.05, smoothing=0.1, maximum=False)

        points_cam = Transformation.transform_points(keyframe.Camera.H_cam_gps, points_gps)

        # Filter points too close to each other in approximated image space (sqrt distance in 3D space)
        previous_point = points_cam[-1]
        for i in range(len(points_cam)-2, 0, -1):
            point = points_cam[i]
            if np.arccos(np.dot(point, previous_point) / (np.linalg.norm(point) * np.linalg.norm(previous_point))) < 0.005:
                points_cam.pop(i)
            else:
                previous_point = points_cam[i]

        points_gps = Transformation.transform_points(keyframe.Camera.H_gps_cam, points_cam)

        # Set color to visualise different tracks
        # color_track = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        color_track = (255, 0, 0) # blue
        color_nodes = (0, 0, 255) # red

        # Separate points into left and right side of track
        points_gps_L, points_gps_R = Transformation.separate_track_into_left_right(points_gps)

        for points_gps_separated in [points_gps_L, points_gps_R]:
            points_cam = Transformation.transform_points(keyframe.Camera.H_cam_gps, points_gps_separated)
            
            pixels = Transformation.project_camera_points_to_pixels(keyframe.Camera, points_cam)

            if len(pixels) < 2:
                continue

            # Visualise results: reprojected tracks + original railway nodes
            reprojected_visual = Visualisation.draw_on_image(reprojected_visual, pixels, False, color_track)

        original_pixels = Transformation.project_points_to_pixels(camera_0, H_cam_w, points_w)
        reprojected_visual = Visualisation.draw_on_image(reprojected_visual, original_pixels, False, color_nodes)

        points_gps_array_track = Transformation.convert_points_list(points_gps_L + points_gps_R, to_type="array")
        points_gps_array = np.append(points_gps_array, points_gps_array_track, axis=0)

    pixels_annotated_array = np.empty((0,2))
        
    for spline, pixel_sequence in zip(keyframe.Annotations.splines, keyframe.Annotations.pixel_sequences):
        reprojected_visual = Visualisation.draw_on_image(reprojected_visual, spline, False, (255, 255, 0)) # cyan
        reprojected_visual = Visualisation.draw_on_image(reprojected_visual, pixel_sequence, False, (0, 255, 255)) # yellow

        pixels_annotated_array_track = Transformation.convert_points_list(spline, to_type="array")
        pixels_annotated_array = np.append(pixels_annotated_array, pixels_annotated_array_track, axis=0)

    print("- # GPS points:", points_gps_array.shape[0])
    print("- # annotated pixels:", pixels_annotated_array.shape[0])

    cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/" + keyframe.filename + "_pre-optimization.jpg", reprojected_visual)

    print("- Saved image pair")


    """
    Optimisation: iterative closest points
    """

    print("Optimising camera pose")

    # Call ceres solver
    new_camera_pose = optimize_camera_pose(keyframe.filename,
                                           np.asarray(keyframe.image),
                                           pixels_annotated_array,
                                           points_gps_array,
                                           keyframe.Camera.intrinsics_vector,
                                           keyframe.Camera.pose_vector)

    assert new_camera_pose.shape == (7,)

    keyframe.Camera.update_pose(new_camera_pose)

print("Finished")