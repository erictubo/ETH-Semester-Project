#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from camera import Camera
from keyframe import KeyFrame
from map_info import MapInfo
from image_features import ImageFeatures
from transformation import Transformation


"""
Global Constants, File Paths & Databases
"""

track_width = 1.435
pole_height = 5.3

path_to_data = '/Users/eric/Developer/Cam2GPS/'
path_to_elevation_data = path_to_data
path_to_osm_file = path_to_data
path_to_pole_location_data = str(path_to_data + 'full_track.csv')
pole_location_dataframe = pd.DataFrame(pd.read_csv(path_to_pole_location_data), columns=['x', 'y'])

path_to_images = str(path_to_data + '/images/')
path_to_poses  = str(path_to_data + '/poses/')

start = 1
stop  = 100
ids = range(1, 100+1)


"""
Camera Parameters
"""
camera = Camera(alpha_u= 698.8258566937119*2, alpha_v= 698.6495855393557*2, u0= 492.9705850660823*2, v0= 293.3927615928415*2)
# optional: define different cameras with different parameters


"""
Main Loop
"""



for id in ids:

    keyframe = KeyFrame(id)


    # Update parameters across keyframes





