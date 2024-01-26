#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# Known constants
track_width = 1.435
# pole_height = 5.3

# Folder containing all data
path_to_data = '/media/psf/Data/'

# Elevation data
path_to_elevation_data = path_to_data + "elevation/"
if not os.path.exists(path_to_elevation_data):
    os.makedirs(path_to_elevation_data)

# Railway map: nodes, tracks
path_to_osm_file = path_to_data + "map/potsdam2.osm"

# Pole locations
# path_to_pole_location_data = str(path_to_data + 'poles.csv')
# pole_location_dataframe = pd.DataFrame(pd.read_csv(path_to_pole_location_data), columns=['x', 'y'])

# Frames: images & poses
# 0: camera 0, 1: camera 1
path_to_frames = str(path_to_data + 'frames/sync/')
path_to_poses  = str(path_to_frames + 'poses/')
path_to_images_0 = str(path_to_frames + 'images_0/')
path_to_images_1 = str(path_to_frames + 'images_1/')
path_to_annotations_0 = path_to_frames + 'annotations_0/annotations.csv'
path_to_annotations_1 = path_to_frames + 'annotations_1/annotations.csv'

railway_object_file = 'railway.pkl'

# Visualization
path_to_visualization = '/media/psf/visualization/'
path_to_visualization_initial = path_to_visualization + 'initial/'
path_to_visualization_optimization = path_to_visualization + 'optimization/'
path_to_visualization_final = path_to_visualization + 'final/'


# check that all visualization paths exist
for path in [path_to_visualization, path_to_visualization_initial, path_to_visualization_optimization, path_to_visualization_final]:
    if not os.path.exists(path):
        os.makedirs(path)