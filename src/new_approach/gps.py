#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation
from math import floor, ceil
import os
import zipfile
import plotly
import urllib.request

from import_osm import (railway_map as RailwayMap, track_node as TrackNode, track_segment as TrackSegment)

from data import path_to_elevation_data, pole_location_dataframe, railway_map
from transformation import Transformation


class GPS:

    """
    Data class that stores information associated with GPS location
    """

    def __init__(self, pose, r_ahead=100, r_behind=50):

        self.is_missing_data = False

        # Extracted from pose: heading, rotation, position, H matrix
        try:
            self.heading = self.__extract_heading__(pose)
            self.position = self.__extract_position__(pose)
            self.quaternions = self.__extract_quaternions__(pose)
        except:
            self.is_missing_data = True
            return

        # GPS in world frame = world frame to GPS
        self.t_w_gps = self.position
        self.x_w_gps = float(self.position[0])
        self.y_w_gps = float(self.position[1])
        self.z_w_gps = float(self.position[2])

        self.R_w_gps = Transformation.convert_quaternions(self.quaternions, to_type='rotation matrix')
        self.H_w_gps = Transformation.compile_H_matrix(self.R_w_gps, self.t_w_gps)
        self.H_gps_w = Transformation.invert_H_matrix(self.H_w_gps)

        # Extracted map information
        self.elevation = MapInfo.get_elevation(self.x_w_gps, self.y_w_gps)

        self.railway_nodes = MapInfo.select_local_railway_nodes(railway_map.railway_nodes, self.x_w_gps, self.y_w_gps, self.heading, r_ahead, r_behind)
        self.railway_nodes_w = MapInfo.convert_railway_nodes_to_world_coordinates(self.railway_nodes)

        self.railway_tracks, self.railway_nodes_in_tracks = MapInfo.get_railway_nodes_by_tracks(self.x_w_gps, self.y_w_gps, self.heading, r_ahead, r_behind)

        self.railway_segment_lists = MapInfo.divide_into_continuous_segments(self.railway_tracks, self.railway_nodes_in_tracks)

    @staticmethod
    def __extract_heading__(pose) -> float:
        heading = pose.get('heading')
        return heading

    @staticmethod
    def __extract_position__(pose) -> np.ndarray[float]:
        position = np.array([[float(pose.get('p_x'))], [float(pose.get('p_y'))], [float(pose.get('p_z'))]])
        return position

    @staticmethod
    def __extract_quaternions__(pose) -> np.ndarray[float]:
        quaternions = Rotation.from_quat([pose["q_x"], pose["q_y"], pose["q_z"], pose["q_w"]])
        return quaternions
            


class MapInfo:
    
    """
    Method class that retrieves data related to GPS location:

    - elevation
    - closest poles
    - railway map
    """

    @staticmethod
    def get_elevation(x_w_gps: float, y_w_gps: float) -> float:
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
    

    """
    Railway Map
    - Nodes
    - Tracks
    """

    # Attributes of railway_map:
        # .railway_nodes
        # .railway_tracks
        # .direct_neighbours_of_nodes
        # .osm_node_ids
        # .osm_track_ids
        # .edge_to_edge_connectivits
        # .segment_to_node_assignment[segment_id]
        # .nodes_to_segment_assignment[node_id]

    @staticmethod
    def plot_railway_map():
        railway_map.plotly(plotly.graph_objs.Figure())

    @staticmethod
    def convert_railway_nodes_to_world_coordinates(nodes: list[TrackNode]) -> list[np.ndarray]:
        world_coordinates = []
        for node in nodes:
            x = node.x
            y = node.y
            z = MapInfo.get_elevation(x, y)
            world_coordinates.append(np.array([[x], [y], [z]]))
        return world_coordinates

    @staticmethod
    def select_local_railway_nodes(nodes: list[TrackNode], p_x: float, p_y: float, heading: float, r_ahead: float, r_behind: float = 0):

        grad_x = np.cos(heading*np.pi/180)
        grad_y = np.sin(heading*np.pi/180)
        
        selected_nodes = []
        for node in nodes:
            dist2 = ((node.x-p_x)**2 + (node.y-p_y)**2)
            
            if dist2 < r_behind**2:
                selected_nodes.append(node)
            
            elif dist2 < r_ahead**2:
                if ((node.x - p_x) * grad_x + (node.y - p_y) * grad_y) > 0:
                    selected_nodes.append(node)
            
        return selected_nodes


    @staticmethod
    def get_railway_nodes_by_tracks(p_x: float, p_y: float, heading: float, r_ahead: float, r_behind: float = 0):

        tracks: list[TrackSegment] = []
        nodes_in_tracks: dict[TrackSegment: list[TrackNode]] = {}

        for track in railway_map.railway_tracks:
            nodes_in_track: list[TrackNode] = []

            # Get all nodes in track
            for node_id in railway_map.segment_to_node_assignment[track.id]:
                node = railway_map.railway_nodes[node_id]
                nodes_in_track.append(node)
            
            # Select nodes according to filters
            nodes_in_track = MapInfo.select_local_railway_nodes(nodes_in_track, p_x, p_y, heading, r_ahead, r_behind)

            if len(nodes_in_track) > 0:
                tracks.append(track)
                nodes_in_tracks[track] = nodes_in_track

        return tracks, nodes_in_tracks
    

    @staticmethod
    def switch_to_railway_tracks_by_nodes(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):

        """
        Output: dictionary with nodes and their tracks, list of nodes that are in multiple tracks
        """
        nodes: list[TrackNode] = []
        tracks_of_nodes: dict[TrackNode: list[TrackSegment]] = []
        nodes_in_multiple_tracks: list[TrackNode] = []

        for track in tracks:
            nodes_in_track = nodes_in_tracks[track]

            for node in nodes_in_track:

                if node not in nodes:
                    nodes.append(node)

                track_ids_of_node = railway_map.nodes_to_segment_assignment[node.id]

                # If node is in multiple tracks
                if len(track_ids_of_node) > 1:
                    tracks_of_node = []
                    # Fill dictionary with all tracks of the node
                    for track_id_of_node in track_ids_of_node:
                        track_of_node = railway_map.railway_tracks[track_id_of_node]
                        tracks_of_node.append(track_of_node)
                    # Append node to dictionary
                    nodes_in_multiple_tracks.append(node)

                else:
                    tracks_of_node = track
                tracks_of_nodes[node] = tracks_of_node

        return nodes, tracks_of_nodes
    

    @staticmethod
    def find_connecting_railway_nodes(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):

        """
        Check if edge nodes of tracks are connecting nodes between multiple tracks.
        """
        edge_nodes: list[TrackNode] = []
        connecting_nodes: list[TrackNode] = []

        for track in tracks:
            nodes_in_track = nodes_in_tracks[track]
            for edge_node in [nodes_in_track[0], nodes_in_track[-1]]:
                if edge_node in edge_nodes:
                    connecting_nodes.append(edge_node)
                edge_nodes.append(edge_node)
        return connecting_nodes


    @staticmethod
    def divide_into_continuous_segments(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):
        """
        Output: list of list of nodes
        """
        segment_lists: list[list[TrackNode]] = []

        for track in tracks:
            nodes_in_track = nodes_in_tracks[track]

            split_index = None

            for i in range(2,len(nodes_in_track)-1):

                previous_node = nodes_in_track[i-1]
                current_node = nodes_in_track[i]

                if current_node.id not in railway_map.direct_neighbours_of_nodes[previous_node.id]:
                    if split_index: print("Error: multiple splits")
                    split_index = i
            if split_index:
                segment_lists.append(nodes_in_track[:split_index])
                segment_lists.append(nodes_in_track[split_index:])
            else:
                segment_lists.append(nodes_in_track)

        return segment_lists
    

    @staticmethod
    def add_neighbours_to_connecting_railway_nodes(segment: list[TrackNode], num_neigbours=1):

        # add neighbours to connecting nodes to improve splines
        # num_neighbours depends on type of spline

        # use transition segments (smallest opposite angle) to decide which node to add


        # # add neighbours of start & end nodes
        # # (points not in image but used for drawing lines)
        # start_node, end_node = nodes_list[0], nodes_list[-1]
        # for node in [start_node, end_node]:
            
        #     # find neighbour_in_nodes_list
        #     for neighbour_index in railway_map.direct_neighbours_of_nodes[node.id]:
        #         neighbour = railway_map.railway_nodes[neighbour_index]
        #         if neighbour in nodes_list:
        #             neighbour_in_nodes_list = neighbour

        #     for neighbour_index in railway_map.direct_neighbours_of_nodes[node.id]:
        #         neighbour = railway_map.railway_nodes[neighbour_index]
        #         if neighbour not in nodes_list:
        #             # check that neighbour is in opposite direction to neighbour_in_list
        #             # (independent of track, for interpolation)
        #             # i.e. dot product negative
        #             if ((neighbour.x-node.x) * (neighbour_in_nodes_list.x-node.x) + \
        #             (neighbour.y-node.y) * (neighbour_in_nodes_list.y-node.y)) < 0:
        #                 if node == start_node: nodes_list.insert(0, neighbour)
        #                 elif node == end_node: nodes_list.append(neighbour)

        return segment


    

