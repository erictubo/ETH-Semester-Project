#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import urllib.request
import zipfile
import plotly

from import_osm import railway_map, track_node, track_segment

from main import path_to_elevation_data, pole_location_dataframe, path_to_osm_file


class MapInfo:
    """
    Class that retrieves data related to GPS Pose:
    - Elevation
    - Closest Poles
    - Map of Rail Network
        - Local Nodes / Tracks
    """


    @staticmethod
    def get_elevation(x_gps_w: float, y_gps_w: float) -> float:
        """
        Retrieves elevation closest to specified GPS position [x, y] \\
        Output: elevation
        """

        file_name = str('dgm_33' + str(x_gps_w)[0:3] + '-' + str(y_gps_w)[0:4])
        local_file_path = str(path_to_elevation_data + file_name)
        
        if not os.path.exists(local_file_path):
            zip_url = str('https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/' + file_name + '.zip' )
            local_path_zip = str(path_to_elevation_data + file_name + '.zip')
            urllib.request.urlretrieve(zip_url, local_path_zip)
            with zipfile.ZipFile(local_path_zip, 'r') as zip_ref:
                zip_ref.extractall(local_file_path)
        
        file = open(str(local_file_path+ '/' + file_name +'.xyz'), 'r')
        min_error = np.inf
        for line in file:
            split_string = line.split()
            x_curr = split_string[0]
            y_curr = split_string[1]
            error = np.abs((x_gps_w - float(x_curr))**2 + (y_gps_w - float(y_curr))**2)
            if error < min_error:
                min_error = error
                elevation = float(split_string[2])
        file.close()

        return elevation
    
    @staticmethod
    def find_closest_pole(elevation: float, H_gps_w: np.ndarray[float]) -> np.ndarray:
        """
        Determines closest pole to the specified GPS position. \\
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
    

    def get_closest_railway_nodes(x, y, radius):

        map = railway_map("Potsdam2").import_from_osm_file(path_to_osm_file + "potsdam2.osm")
        
        # Attributes of railway_map:
        # .railway_nodes
        # .railway_tracks
        # .direct_neighbours_of_nodes
        # .osm_node_ids
        # .osm_track_ids
        # .edge_to_edge_connectivits
        # .segment_to_node_assignment
        # .nodes_to_segment_assignment

        # map.plotly(plotly.graph_objs.Figure())

        for i, node in enumerate(self.map.railway_nodes):
            #print(i, node.lat, node.lon, node.x, node.y)

            #print(x-node.x, y-node.y)

            if ((node.x - x)**2 + (node.y - y)**2)**0.5 < radius:
                print(i, abs(node.x - x), abs(node.y-y))




x = 373509.8500102493
y = 5804179.834478266

MapInfo.get_closest_railway_nodes(x,y,1500)


# distance too large: at least 1.5 km
# three possible reasons
# 1. different coordinate systems
# 2. osm files don't correspond to images
# 3. error in poses