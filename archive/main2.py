#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External Libraries
import numpy as np
import cv2
import random
import pickle
import matplotlib.pyplot as plt

# Methods & Data
from data import railway_object_file
from camera import Camera
from keyframe import KeyFrame

from railway import Railway
from transformation import Transformation
from visualisation import Visualisation


""" Camera Parameters """
Camera_0 = Camera(image_size = (1920, 1200),
                  focal_length = [2*698.8258566937119, 2*698.6495855393557], 
                  principal_point = [2*492.9705850660823, 2*293.3927615928415], 
                  distortion_coefficients = [0.15677829213401465, 0.1193983032005652, -0.011287926707786692, 0.23013426724923478]
                  )

Camera_1 = Camera(image_size = (1920, 1200),
                  focal_length = [2*698.5551622016458, 2*698.6714179616163],
                  principal_point = [2*489.7761220331873, 2*297.9103051604121],
                  distortion_coefficients = [0.16213733886648546, 0.11615082942281711, -0.05779296718987261, 0.324051803251489]
                  )


""" Keyframes """
def create_keyframes(ids):
    print("Creating keyframes")
    keyframes: list[KeyFrame] = []
    for id in ids:
        id = int(id)
        keyframe = KeyFrame(id, Camera_0)
        if keyframe.GPS.is_missing_data:
            # print("Keyframe #", id, "missing data >> skipped")
            np.delete(ids, np.where(ids == id))
            continue
        # print("Keyframe #", id)
        keyframes.append(keyframe)
    return keyframes


ids = np.arange(0, 1928, 25)

"""
Pre-processing of relevant railway map:
- fill gaps
- elevation
"""

max_gap = 20
r_ahead = 100
r_behind = 50


create_railway = input("Create a new railway object? [y/n]: ")
if create_railway == 'y':

    keyframes = create_keyframes(ids)

    railway = Railway(keyframes, max_gap=max_gap, r_ahead=r_ahead, r_behind=r_behind)

    with open(railway_object_file, 'wb') as file:
        pickle.dump(railway, file)
        print("Railway object sucessfully saved to file.")

else:
    print("Loading railway object from file")
    with open(railway_object_file, 'rb') as file:
        railway = pickle.load(file)


plot_railway_2D = False
plot_railway_3D = False

if plot_railway_2D:
    keyframes = create_keyframes(ids)

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

if plot_railway_3D:
    keyframes = create_keyframes(ids)

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


# Camera frame in GPS frame
R_gps_camf = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
t_gps_camf = np.array([[0],[0],[0]])                                    
H_gps_camf = Transformation.compile_H_matrix(R_gps_camf, t_gps_camf)


"""
Offset: initial guess
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


ids = [325, 450, 600, 775, 1025, 1225, 1650, 1800, 1900]
keyframes = create_keyframes(ids)

for keyframe in keyframes:

    H_w_gps = keyframe.GPS.H_w_gps
    H_gps_w = keyframe.GPS.H_gps_w

    H_gps_cam = H_gps_camf @ H_camf_cam
    H_cam_gps = Transformation.invert_H_matrix(H_gps_cam)

    H_w_cam = H_w_gps @ H_gps_cam
    H_cam_w = Transformation.invert_H_matrix(H_w_cam)


    """
    Reproject local railway onto image
    - 3D interpolation
    - interpolation in image space
    """

    reprojected_visual = keyframe.image.copy()
    
    local_tracks, local_points_in_tracks = railway.get_local_points_in_tracks(keyframe, r_ahead, r_behind, min_points=2)

    for track in local_tracks:

        points_w = local_points_in_tracks[track]
        
        points_gps = Transformation.transform_points(H_gps_w, points_w)

        points_gps = Transformation.interpolate_spline_linspace(points_gps, desired_spacing=0.05, smoothing=0.1, maximum=False)

        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        points_gps_L, points_gps_R = Transformation.separate_track_into_left_right(points_gps)

        for points_gps_separated in [points_gps_L, points_gps_R]:

            # Convert to camera frame, using prior estimate of camera pose
            points_cam = Transformation.transform_points(H_cam_gps, points_gps_separated)

            # Reproject onto image plane
            pixels = Transformation.project_camera_points_to_pixels(keyframe.Camera, points_cam)

            if len(pixels) > 2:
                unique_pixels = [pixels[0]]
                for pixel in pixels:
                    if np.linalg.norm(pixel - unique_pixels[-1]) >= 10:
                        unique_pixels.append(pixel)
                pixels = unique_pixels

                # lines = Visualisation.convert_consecutive_pixels_to_2D_lines(pixels)
                reprojected_visual = Visualisation.draw_on_image(reprojected_visual, pixels, False, (255, 0, 0))

                original_pixels = Transformation.project_points_to_pixels(Camera_0, H_cam_w, points_w)
                reprojected_visual = Visualisation.draw_on_image(reprojected_visual, original_pixels, False, (0,0,255))


    annotated_visual = reprojected_visual.copy()
        
    for spline, pixel_sequence in zip(keyframe.Annotations.splines, keyframe.Annotations.pixel_sequences):
        annotated_visual = Visualisation.draw_on_image(annotated_visual, spline, False, (255, 255, 0))
        annotated_visual = Visualisation.draw_on_image(annotated_visual, pixel_sequence, False, (0, 255, 255))


    cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/reprojected splines undistorted/" + keyframe.filename + "_reprojection.jpg", reprojected_visual)
    cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/annotated splines undistorted/" + keyframe.filename + "_annotation.jpg", annotated_visual)

    print("Saved image pair")

print("Finished")