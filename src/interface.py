
import sys
import glob
import numpy as np
from pathlib import Path

sys.path.append('./cpp')

from cpp.optimization import optimize_camera_pose


if __name__ == '__main__':

    # 3D points
    gps_3D_points = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [1, 1, 0],
                              [0, 1, 0]], dtype=np.float32)

    # 2D points
    observed_2D_points = np.array([[0, 0],
                                   [1, 0],
                                   [1, 1],
                                   [0, 1]], dtype=np.float32)

    # Camera matrix
    camera_intrinsics = np.array([1400.0, 1400.0, 1000.0, 600.0], dtype=np.float32)
    camera_pose = np.array([0.1, 0.1, 0.1, 0.5, -0.5, 0.5, -0.5], dtype=np.float32)

    print(camera_intrinsics.shape)
    print(camera_pose.shape)

    # Optimize camera pose
    new_camera_pose = optimize_camera_pose(camera_intrinsics, camera_pose, observed_2D_points, gps_3D_points)

    #print("New camera pose: ", new_camera_pose)
