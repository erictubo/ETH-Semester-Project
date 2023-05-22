#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External libraries
import numpy as np
from math import floor, ceil
import os
import zipfile
import urllib.request

# Data & methods
from data import path_to_elevation_data, pole_location_dataframe


class MapInfo:
    
    """
    Methods for retrieving:
    - elevation
    - poles
    """

    @staticmethod
    def get_elevation(x_w_gps: float, y_w_gps: float):
        """
        Retrieves elevation closest to specified GPS position [x, y]
        Output: elevation
        """
        x = round(x_w_gps, 1)
        y = round(y_w_gps, 1)

        file_name = str('dgm_33' + str(x)[0:3] + '-' + str(y)[0:4])
        local_file_path = str(path_to_elevation_data + file_name)
        
        if not os.path.exists(local_file_path):
            zip_url = str('https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/' + file_name + '.zip' )
            local_path_zip = str(path_to_elevation_data + file_name + '.zip')
            urllib.request.urlretrieve(zip_url, local_path_zip)
            with zipfile.ZipFile(local_path_zip, 'r') as zip_ref:
                zip_ref.extractall(local_file_path)
        
        # print("Searching elevation at coordinates:", str([x,y]))

        file = open(str(local_file_path+ '/' + file_name +'.xyz'), 'r')
        lines = file.readlines()

        idx_y1, idx_y2 = 1000, 1
        y1 = float(lines[1000*idx_y1-1].split(",")[1])
        y2 = float(lines[1000*idx_y2-1].split(",")[1])

        # 1000 different y-values, each for 1000 lines
        while y2 != y1:
            if abs(y-y1) < abs(y-y2):
                idx_y2 = ceil((idx_y1+idx_y2)/2)
                y2 = float(lines[1000*idx_y2-1].split(",")[1])
            else:
                idx_y1 = floor((idx_y1+idx_y2)/2)
                y1 = float(lines[1000*idx_y1-1].split(",")[1])
            # print("y:", y, " | y1, y2:", y1, y2, " | idx_y1, idx_y2:", idx_y1, idx_y2)
        assert idx_y1 == idx_y2

        # 1000 different x-values, all with same y-value
        idx_x2 = 1000*idx_y2
        idx_x1 = idx_x2 - 999

        x1 = float(lines[idx_x1-1].split(",")[0])
        x2 = float(lines[idx_x2-1].split(",")[0])

        while x2 != x1:
            if abs(x-x1) < abs(x-x2):
                idx_x2 = floor((idx_x1+idx_x2)/2)
                x2 = float(lines[idx_x2-1].split(",")[0])
            else:
                idx_x1 = ceil((idx_x1+idx_x2)/2)
                x1 = float(lines[idx_x1-1].split(",")[0])
            # print("x:", x, " | x1, x2:", x1, x2, " | idx_x1, idx_x2:", idx_x1, idx_x2)
        assert idx_x1 == idx_x2
        elevation = float(lines[idx_x1-1].split(",")[2])
        file.close()

        # print("    Found elevation at coordinates:", str([x1,y1]), "of:", elevation)

        return elevation
    

    @staticmethod
    def find_closest_pole(elevation: float, H_gps_w: np.ndarray[float]) -> np.ndarray:
        """
        Determines closest pole to the specified GPS position.
        Output: pole_worldview_gps
        """

        df = pole_location_dataframe
        for index, pole in df.iterrows():
            test_pole_gps_world_view = np.array([[df.x[index]], [df.y[index]], [elevation]])
            test_pole_gps = ( H_gps_w @ np.r_[test_pole_gps_world_view, [[1]]] )[0:3,:]
            if smallest_diff > test_pole_gps[0] > 0:
                smallest_diff = test_pole_gps[0]
                smallest_index = index

        x_pole_world = df.x[smallest_index]
        y_pole_world = df.y[smallest_index]
        pole_elevation = MapInfo.get_elevation(x_pole_world, y_pole_world)

        pole_worldview_world = np.array([[x_pole_world], [x_pole_world], [pole_elevation]])
        pole_worldview_gps = ( H_gps_w @ np.r_[pole_worldview_world, [[1]]] )[0:3,:]

        return pole_worldview_gps

    # ToDo: multiple closest poles