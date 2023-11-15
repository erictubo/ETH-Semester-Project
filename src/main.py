#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External Libraries
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Methods & Data
import data
from camera import Camera
from railway import Railway

from keyframe import Frame, Keyframe
from transformation import Transformation

import cpp.optimization


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

offset_camf_cam = [0., -1.5, -8.]
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

camera_0 = Camera(0, image_size = (1920, 1200),
                  focal_length = [2*698.8258566937119, 2*698.6495855393557], 
                  principal_point = [2*492.9705850660823, 2*293.3927615928415], 
                  distortion_coefficients = [0.15677829213401465, 0.1193983032005652, -0.011287926707786692, 0.23013426724923478],
                  initial_H_gps_cam=initial_H_gps_cam)

camera_1 = Camera(1, image_size = (1920, 1200),
                  focal_length = [2*698.5551622016458, 2*698.6714179616163],
                  principal_point = [2*489.7761220331873, 2*297.9103051604121],
                  distortion_coefficients = [0.16213733886648546, 0.11615082942281711, -0.05779296718987261, 0.324051803251489],
                  initial_H_gps_cam=initial_H_gps_cam)

cameras = [camera_0, camera_1]


def create_frames(ids: list[int], include_elevation=True) -> list[Frame]:
    """
    Create a list of basic frames for processing of the Railway object.
    """
    frames: list[Frame] = []
    for id in ids:
        id = int(id)
        frame = Frame(id, include_elevation)
        if frame.gps.is_missing_data:
            print("Frame #", id, "missing data >> skipped")
            np.delete(ids, np.where(ids == id))
            continue
        print("Frame #", id, "created")
        frames.append(frame)
    return frames


def create_keyframes(ids: list[int], camera_0: Camera, camera_1: Camera, railway: Railway) -> list[Keyframe]:
    """
    Create a list of sophisticated keyframes from the same camera for optimisation.
    Requires an existing Railway object (to create an attribute of local points).
    """
    keyframes: list[Keyframe] = []
    for id in ids:
        id = int(id)
        keyframe = Keyframe(id, camera_0, camera_1, railway)
        if keyframe.gps.is_missing_data:
            print("Keyframe #", id, "missing data >> skipped")
            np.delete(ids, np.where(ids == id))
            continue
        print("Keyframe #", id, "created")
        keyframes.append(keyframe)
    return keyframes

"""
Dense frames for Railway processing
"""
frame_ids = np.arange(0, 5495, 10)
frame_ids = np.arange(4000, 4500, 1)

"""
Railway object: process new (and save to file) or load existing
"""
max_gap = 20
r_ahead = 100
r_behind = 50

if (not os.path.isfile(data.railway_object_file)) or (input("Create a new railway object? [y/n]: ") == 'y'):
    frames = create_frames(frame_ids, include_elevation=False)
    railway = Railway(frames, max_gap=max_gap, r_ahead=r_ahead, r_behind=r_behind)
    with open(data.railway_object_file, 'wb') as file:
        pickle.dump(railway, file)
        print("Railway object sucessfully saved to file.")
else:
    print("Loading railway object from file")
    with open(data.railway_object_file, 'rb') as file:
        railway = pickle.load(file)


# railway.plot_map()

"""
Visualization of pre-processed railway in 2D and 3D
"""
if input("Visualize railway / frames? [y/n]: ") == 'y':
    if input("2D? [y/n] ") == 'y':
        frames = create_frames(frame_ids, include_elevation=False)
        # railway.visualise_2D(show_tracks=False, frames=[])
        # railway.visualise_2D(show_tracks=True, frames=[])
        railway.visualise_2D(show_tracks=True, frames=frames)

    if input("GPS Height vs. Elevation? [y/n] ") == 'y':
        frames = create_frames(frame_ids, include_elevation=True)
        # for frame in frames:
        #     point = frame.gps.t_w_gps
        #     plt.scatter(frame.id, point[2], s=3, c='black')
        #     plt.scatter(frame.id, frame.gps.elevation, s=3, c='red')
        #     plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
        #     plt.xticks(np.arange(0, 5500, 500))
        #     plt.yticks(np.arange(30, 45, 1))
        #     plt.xlabel("Frame ID")
        #     plt.ylabel("Height [m]")
        #     plt.legend(["GPS height", "Local elevation"])
        # plt.show()
        
        for frame in frames:
            point = frame.gps.t_w_gps
            plt.scatter(frame.id, point[2], s=3, c='blue')
            plt.scatter(frame.id, frame.gps.elevation, s=3, c='red')
            plt.xticks(np.arange(4000, 4500, 100))
            # plt.yticks(np.arange(33, 37, 0.2))
            plt.xlabel("Frame ID")
            plt.ylabel("Height [m]")
            plt.legend(["GPS height", "Local elevation"])
        plt.show()
    
    if input("GPS Rotation? [y/n] ") == 'y':
        frames = create_frames(frame_ids, include_elevation=False)

        yaw_0 = frames[0].gps.angles_deg[0]
        pitch_0 = frames[0].gps.angles_deg[1]
        roll_0 = frames[0].gps.angles_deg[2]

        yaw_0 = 0
        pitch_0 = 0
        roll_0 = 0

        for frame in frames:
            yaw = frame.gps.angles_deg[0] - yaw_0
            pitch = frame.gps.angles_deg[1] - pitch_0
            roll = frame.gps.angles_deg[2] - roll_0

            plt.scatter(frame.id, roll, s=3, c='red')
            plt.scatter(frame.id, pitch, s=3, c='green')
            plt.scatter(frame.id, yaw, s=3, c='blue')
            # plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            plt.xlabel("Frame ID")
            plt.ylabel("Rotation [deg]")
            plt.legend(["Roll", "Pitch", "Yaw"])
        plt.show()

    if input("3D? [y/n] ") == 'y':
        frames = create_frames(frame_ids, camera_0, include_elevation=True)
        railway.visualise_3D(frames)

    if input("Continue to optimize camera pose? [y/n]: ") == 'n':
        exit()


"""
Sparse keyframes for optimization -- subset of prior frames
"""
# keyframe_ids = [780, 1090, 1430, 2770, 3110, 4500, 5410, 5420, 5430, 5440, 5450, 5460, 5470, 5480]
# keyframe_ids = [780, 3110, 5410]
# keyframe_ids = [5410, 5420, 5430, 5440, 5450, 5460, 5470, 5480]
# keyframe_ids = [5410, 5420, 5430]
# keyframe_ids = [5410, 5420, 5430]
# keyframe_ids = [780, 1090, 1430]
# keyframe_ids = [490, 570, 950]
# keyframe_ids = [570, 950]
# keyframe_ids = [5410]
keyframe_ids: list[int] = [950]

# for keyframe_id in keyframe_ids:
#     assert keyframe_id in frame_ids, "Keyframe ID not in frame IDs"

keyframes = create_keyframes(keyframe_ids, camera_0, camera_1, railway)

"""
Initial reprojection of keyframes
"""
print("Reprojecting points using initial camera pose")

for i, keyframe in enumerate(keyframes):

    annotated_visual_0 = keyframe.annotation_0.visualise_splines_and_points()
    annotated_visual_1 = keyframe.annotation_1.visualise_splines_and_points()

    cv2.imwrite(data.path_to_visualization_initial + keyframe.filename + "_annotated_0.jpg", annotated_visual_0)
    cv2.imwrite(data.path_to_visualization_initial + keyframe.filename + "_annotated_1.jpg", annotated_visual_1)

    reprojected_visual_0 = keyframe.visualise_reprojected_and_original_points(camera_0)
    reprojected_visual_1 = keyframe.visualise_reprojected_and_original_points(camera_1)
    
    cv2.imwrite(data.path_to_visualization_initial + keyframe.filename + "_reprojected_0.jpg", reprojected_visual_0)
    cv2.imwrite(data.path_to_visualization_initial + keyframe.filename + "_reprojected_1.jpg", reprojected_visual_1)

    combined_visual_0 = keyframe.visualise_reprojected_points(camera_0, keyframe.annotation_0.visualise_splines())
    combined_visual_1 = keyframe.visualise_reprojected_points(camera_1, keyframe.annotation_1.visualise_splines())

    cv2.imwrite(data.path_to_visualization_initial + keyframe.filename + "_combined_0.jpg", combined_visual_0)
    cv2.imwrite(data.path_to_visualization_initial + keyframe.filename + "_combined_1.jpg", combined_visual_1)


"""
Optimization: iterative closest points
"""
print("Optimizing camera pose")

iterations = 10

"""
Camera 0
"""

cpp.optimization.reset_keyframes()

for keyframe in keyframes:
    assert keyframe.camera_0 == camera_0
    cpp.optimization.add_keyframe(
        keyframe.filename,
        str(keyframe.camera_0.id),
        np.asarray(keyframe.image_0),
        keyframe.annotation_0.array,
        keyframe.points_gps_array_0)

new_camera_pose = cpp.optimization.update_camera_pose(camera_0.pose_vector, camera_0.intrinsics_vector, iterations, data.path_to_visualization_optimization)
camera_0.update_pose(new_camera_pose)


"""
Camera 1
"""

cpp.optimization.reset_keyframes()

for keyframe in keyframes:
    assert keyframe.camera_1 == camera_1
    cpp.optimization.add_keyframe(
        keyframe.filename,
        str(keyframe.camera_1.id),
        np.asarray(keyframe.image_1),
        keyframe.annotation_1.array,
        keyframe.points_gps_array_1)
    
new_camera_pose = cpp.optimization.update_camera_pose(camera_1.pose_vector, camera_1.intrinsics_vector, iterations, data.path_to_visualization_optimization)
camera_1.update_pose(new_camera_pose)


for camera in [camera_0, camera_1]:
    print("Final pose of camera #", camera.id, ":\n")
    print("R_gps_cam:\n", camera.H_gps_cam[0:3, 0:3])
    print("t_gps_cam:\n", camera.H_gps_cam[0:3, 3])
    print("q_gps_cam:\n", Transformation.convert_rotation_matrix(camera.H_gps_cam[0:3, 0:3], to_type='quaternion'))


"""
Transformation between cameras
"""

print("Transformation between stereo cameras:\n")
H_cam1_cam0 = camera_1.H_cam_gps @ camera_0.H_gps_cam
R_cam1_cam0, t_cam1_cam0 = Transformation.separate_homogeneous_transformation(H_cam1_cam0)
q_cam1_cam0 = Transformation.convert_rotation_matrix(R_cam1_cam0, to_type='quaternion')
print("R_cam1_cam0:\n", R_cam1_cam0)
print("t_cam1_cam0:\n", t_cam1_cam0)
print("q_cam1_cam0:\n", q_cam1_cam0)


"""
Given calibration between cameras
"""

print("Given calibration between stereo cameras")
t_1_0 = np.array([0.30748946, 0.00190947, 0.00966052])
q_1_0 = np.array([0.00666986, 0.02121604, -0.00106887, 0.99975209])
R_1_0 = Transformation.convert_quaternions(q_1_0, to_type='rotation matrix')
print("R_cam1_cam0:\n", R_1_0)
print("t_cam1_cam0:\n", t_1_0)
print("q_cam1_cam0:\n", q_1_0)

# Absolute position error

print("Absolute position error:\n")
t_error = t_cam1_cam0 - t_1_0
print("t_error:\n", t_error)


"""
Final reprojection of keyframes
"""
print("Reprojecting points using final camera pose")

for i, keyframe in enumerate(keyframes):

    # annotated_visual_0 = keyframe.annotation_0.visualise_splines_and_points()
    # annotated_visual_1 = keyframe.annotation_1.visualise_splines_and_points()

    # cv2.imwrite(data.path_to_visualization_final + keyframe.filename + "_annotated_0.jpg", annotated_visual_0)
    # cv2.imwrite(data.path_to_visualization_final + keyframe.filename + "_annotated_1.jpg", annotated_visual_1)

    # reprojected_visual_0 = keyframe.visualise_reprojected_and_original_points(camera_0)
    # reprojected_visual_1 = keyframe.visualise_reprojected_and_original_points(camera_1)
    
    # cv2.imwrite(data.path_to_visualization_final + keyframe.filename + "_reprojected_0.jpg", reprojected_visual_0)
    # cv2.imwrite(data.path_to_visualization_final + keyframe.filename + "_reprojected_1.jpg", reprojected_visual_1)

    combined_visual_0 = keyframe.visualise_reprojected_points(camera_0, keyframe.annotation_0.visualise_splines())
    combined_visual_1 = keyframe.visualise_reprojected_points(camera_1, keyframe.annotation_1.visualise_splines())

    cv2.imwrite(data.path_to_visualization_final + keyframe.filename + "_combined_0.jpg", combined_visual_0)
    cv2.imwrite(data.path_to_visualization_final + keyframe.filename + "_combined_1.jpg", combined_visual_1)