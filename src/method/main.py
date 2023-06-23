#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External Libraries
import os
import sys
import pickle
import numpy as np
import cv2

# Methods & Data
from data import railway_object_file
from camera import Camera
from railway import Railway

from keyframe import Frame, Keyframe
from transformation import Transformation

src_path = os.path.abspath('../LROD_extrinsics_estimation/src')
sys.path.append(src_path)
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

offset_camf_cam = [0., -2., -10.]
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

camera_1 = Camera(1, image_size = (1920, 1200),
                  focal_length = [2*698.8258566937119, 2*698.6495855393557], 
                  principal_point = [2*492.9705850660823, 2*293.3927615928415], 
                  distortion_coefficients = [0.15677829213401465, 0.1193983032005652, -0.011287926707786692, 0.23013426724923478],
                  initial_H_gps_cam=initial_H_gps_cam)

# camera_2 = Camera(2, image_size = (1920, 1200),
#                   focal_length = [2*698.5551622016458, 2*698.6714179616163],
#                   principal_point = [2*489.7761220331873, 2*297.9103051604121],
#                   distortion_coefficients = [0.16213733886648546, 0.11615082942281711, -0.05779296718987261, 0.324051803251489],
#                   initial_H_gps_cam=initial_H_gps_cam_1
#                   )


def create_frames(ids: list[int]) -> list[Frame]:
    """
    Create a list of basic frames for processing of the Railway object.
    """
    frames: list[Frame] = []
    for id in ids:
        id = int(id)
        frame = Frame(id)
        if frame.gps.is_missing_data:
            print("Frame #", id, "missing data >> skipped")
            np.delete(ids, np.where(ids == id))
            continue
        # print("Frame #", id)
        # print("Frame", id, "added")
        frames.append(frame)
    return frames


def create_keyframes(ids: list[int], camera: Camera, railway: Railway) -> list[Keyframe]:
    """
    Create a list of sophisticated keyframes from the same camera for optimisation.
    Requires an existing Railway object (to create an attribute of local points).
    """
    keyframes: list[Keyframe] = []
    for id in ids:
        id = int(id)
        keyframe = Keyframe(id, camera, railway)
        if keyframe.gps.is_missing_data:
            print("Keyframe #", id, "missing data >> skipped")
            np.delete(ids, np.where(ids == id))
            continue
        keyframes.append(keyframe)
    return keyframes

"""
Dense frames for Railway processing
"""
frame_ids = np.arange(0, 1928, 25)

"""
Railway object: process new (and save to file) or load existing
"""
max_gap = 20
r_ahead = 100
r_behind = 50

if (not os.path.isfile(railway_object_file)) or (input("Create a new railway object? [y/n]: ") == 'y'):
    frames = create_frames(frame_ids)
    railway = Railway(frames, max_gap=max_gap, r_ahead=r_ahead, r_behind=r_behind)
    with open(railway_object_file, 'wb') as file:
        pickle.dump(railway, file)
        print("Railway object sucessfully saved to file.")
else:
    print("Loading railway object from file")
    with open(railway_object_file, 'rb') as file:
        railway = pickle.load(file)


"""
Visualization of pre-processed railway in 2D and 3D
"""
if input("Visualize railway? [y/n]: ") == 'y':
    if not frames:
        frames = create_frames(frame_ids)
    if input("2D? [y/n] ") == 'y':
        railway.visualise_2D(frames)
    if input("3D? [y/n] ") == 'y':
        railway.visualise_3D(frames)


"""
Sparse keyframes for optimization -- subset of prior frames
"""
keyframe_ids = [325, 450, 600, 775, 1025, 1225, 1650, 1800, 1900]
# keyframe_ids = [1900, 775, 1800, 1650, 600]

for keyframe_id in keyframe_ids:
    assert keyframe_id in frame_ids, "Keyframe ID not in frame IDs"

keyframes = create_keyframes(keyframe_ids, camera_1, railway)


"""
Optimization: iterative closest points
"""
print("Optimizing camera pose")

for camera in [camera_1]:

    cpp.optimization.reset_keyframes()

    for keyframe in keyframes:
        assert keyframe.camera == camera, "Keyframe camera does not match camera being optimized"

        cpp.optimization.add_keyframe(
            keyframe.filename,
            np.asarray(keyframe.image),
            keyframe.annotations.array,
            keyframe.points_gps_array)

    new_camera_pose = cpp.optimization.update_camera_pose(camera.pose_vector, camera.intrinsics_vector)

    camera.update_pose(new_camera_pose)

    print("Final pose of camera", camera.id, ":")
    print("Pose vector [t_cam_gps, quaternion]:\n", camera.pose_vector)
    print("R_gps_cam:\n", camera.H_gps_cam[0:3, 0:3])
    print("t_gps_cam:\n", camera.H_gps_cam[0:3, 3])


"""
Final reprojection on all keyframes
"""
print("Reprojecting points using final camera pose")

for i, keyframe in enumerate(keyframes):

    annotated_visual = keyframe.annotations.visualise()
    original_visual = keyframe.visualise_original_reprojected_points(annotated_visual)
    reprojected_visual = keyframe.visualise_reprojected_points(original_visual)
    
    cv2.imwrite("/Users/eric/Developer/Cam2GPS/visualisation/" + keyframe.filename + "_final.jpg", reprojected_visual)


"""
Evaluation metrics
"""
print("Evaluating performance metrics")

# TODO: evaluate performance metrics (reprojection error, etc.) per camera
# TODO: compare to known transformation between stereo camera setup