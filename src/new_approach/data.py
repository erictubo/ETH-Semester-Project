#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from import_osm import (railway_map as RailwayMap, track_node as TrackNode, track_segment as TrackSegment)

"""
Global constants, file paths & databases
"""

track_width = 1.435
pole_height = 5.3

path_to_data = '/Users/eric/Developer/Cam2GPS/'
path_to_elevation_data = path_to_data
path_to_osm_file = path_to_data
path_to_pole_location_data = str(path_to_data + 'full_track.csv')
pole_location_dataframe = pd.DataFrame(pd.read_csv(path_to_pole_location_data), columns=['x', 'y'])
railway_map = RailwayMap("Potsdam2")
railway_map.import_from_osm_file(path_to_osm_file + "potsdam2.osm")


path_to_images = str(path_to_data + '/images/')
path_to_poses  = str(path_to_data + '/poses/')