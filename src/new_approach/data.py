#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from import_osm import railway_map


""" Global Constants """

track_width = 1.435
pole_height = 5.3

# Folder containing all data
path_to_data = '/Users/eric/Developer/Cam2GPS/'

# Elevation data
path_to_elevation_data = path_to_data +"elevation/"

# Railway map: nodes, tracks
path_to_osm_file = path_to_data
railway_map = railway_map("Potsdam2")
railway_map.import_from_osm_file(path_to_osm_file + "potsdam2.osm")

# Pole locations
path_to_pole_location_data = str(path_to_data + 'poles.csv')
pole_location_dataframe = pd.DataFrame(pd.read_csv(path_to_pole_location_data), columns=['x', 'y'])

# Frames: images & poses
path_to_frames = str(path_to_data + 'frames/1929/')
path_to_images = str(path_to_frames + 'images/')
path_to_poses  = str(path_to_frames + 'poses/')