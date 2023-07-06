#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


# Known constants
track_width = 1.435
pole_height = 5.3

# Folder containing all data
path_to_data = '/Users/eric/Developer/Cam2GPS/'

# Elevation data
path_to_elevation_data = path_to_data + "elevation/"

# Railway map: nodes, tracks
path_to_osm_file = path_to_data + "map/"

# Pole locations
path_to_pole_location_data = str(path_to_data + 'poles.csv')
pole_location_dataframe = pd.DataFrame(pd.read_csv(path_to_pole_location_data), columns=['x', 'y'])

# Frames: images & poses
path_to_frames = str(path_to_data + 'frames/2462/')
path_to_poses  = str(path_to_frames + 'poses/')
path_to_images_0 = str(path_to_frames + 'images_0/')
path_to_images_1 = str(path_to_frames + 'images_1/')
path_to_annotations_0 = path_to_frames + 'annotations_0/'
path_to_annotations_1 = path_to_frames + 'annotations_1/'

railway_object_file = 'railway.pkl'