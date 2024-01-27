#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes and manages railway map data for the camera localization pipeline.

Handles extraction, interpolation, and elevation assignment for railway tracks using OSM and elevation data. Provides visualization utilities for 2D/3D railway maps.
"""

# External libraries
import plotly
import numpy as np
import math
import matplotlib.pyplot as plt

# Data & methods
from data import path_to_osm_file
from map_info import MapInfo
from visualization import Visualization

# OSM railway data processing (C. von Einem, ETH Zurich)
from import_osm import (railway_map as RailwayMap, track_node as TrackNode, track_segment as TrackSegment)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gps import GPS
    from keyframe import Frame
    


"""
Attributes of railway_map:
    .railway_nodes
    .railway_tracks
    .direct_neighbours_of_nodes
    .osm_node_ids
    .osm_track_ids
    .edge_to_edge_connectivits
    .segment_to_node_assignment[segment_id]
    .nodes_to_segment_assignment[node_id]
"""


class Railway:
    """
    Represents a processed railway in the combined map area of given keyframes.
    Imports relevant railway nodes, interpolates gaps, adds elevation data, and creates compatible tracks.
    """

    def __init__(self, frames: list['Frame'], max_gap: float, r_ahead: float, r_behind: float):
        """
        Initialize a Railway object from a list of frames and parameters.
        Args:
            frames: List of Frame objects.
            max_gap: Maximum allowed gap between points for interpolation (meters).
            r_ahead: Search radius ahead of each frame (meters).
            r_behind: Search radius behind each frame (meters).
        """
        self.max_gap = max_gap
        self.r_ahead = r_ahead
        self.r_behind = r_behind

        self.railway_map = RailwayMap("Potsdam2")
        self.railway_map.import_from_osm_file(path_to_osm_file)

        print("Extracting relevant nodes and tracks")
        self.nodes = self.__get_relevant_nodes__(frames, r_ahead, r_behind)
        self.tracks, self.tracks_of_nodes = self.__get_tracks_of_nodes__(self.nodes)
        self.nodes_in_tracks = self.__get_nodes_in_tracks__(self.tracks, self.nodes)

        print("Found", len(self.tracks), "relevant tracks, with a total of", len(self.nodes), "relevant nodes.")

        print("Filling railway gaps using max_gap =", max_gap, "[m] ...")
        self.points_in_tracks_2D = self.__convert_nodes_to_gapless_2D_points_in_tracks__(self.tracks, self.nodes_in_tracks, max_gap)

        total_points = 0
        for track in self.tracks:
            total_points += len(self.points_in_tracks_2D[track])
        print("Total points:", total_points)

        print("Adding elevation to points...")
        self.points_in_tracks_3D = self.__convert_2D_to_3D_points_in_tracks__(self.tracks, self.points_in_tracks_2D)      


    def plot_map(self):
        """
        Visualize the railway map using Plotly.
        """
        self.railway_map.plotly(plotly.graph_objs.Figure())

    
    def visualize_2D(self, show_tracks=True, show_nodes=True, frames: list['Frame'] = []):
        """
        Visualize the 2D railway map, nodes, and optionally frames.
        Args:
            show_tracks: Whether to show tracks (default: True).
            show_nodes: Whether to show nodes (default: True).
            frames: List of Frame objects to plot (default: empty).
        """
        if show_tracks:
            for track in self.tracks:
                for point in self.points_in_tracks_2D[track]:
                    Visualization.plot_XY(point[0], point[1], 'orange')
        if show_nodes:
            for node in self.nodes:
                Visualization.plot_XY(node.x, node.y, color='red')
        if len(frames) > 0:
            for frame in frames:
                point = frame.gps.t_w_gps
                Visualization.plot_XY(point[0], point[1], 'black')
                #plt.Circle((point[0], point[1]), r_ahead, 'red')
        plt.xlabel("Position X [m]")
        plt.ylabel("Position Y [m]")
        Visualization.show_plot()

    def visualize_3D(self, frames: list['Frame'] = []):
        """
        Visualize the 3D railway map and optionally frames.
        Args:
            frames: List of Frame objects to plot (default: empty).
        """
        ax = Visualization.create_3D_plot("All points in tracks")
        for track in self.tracks:
            color = 'blue'
            if track.is_bridge:
                color = 'cyan'
            points = self.points_in_tracks_3D[track]
            Visualization.plot_3D_points(ax, points, color)
        for frame in frames:
            point = frame.gps.t_w_gps
            Visualization.plot_3D_points(ax, [point], 'red')
        Visualization.show_plot()


    # Hidden methods: called at initialisation

    def __get_relevant_nodes__(self, frames: list['Frame'], r_ahead: float, r_behind: float):
        """
        Get relevant railway nodes within a search radius of each frame.
        Args:
            frames: List of Frame objects.
            r_ahead: Search radius ahead (meters).
            r_behind: Search radius behind (meters).
        Returns:
            List of relevant TrackNode objects.
        """
        nodes: list[TrackNode] = []
        for frame in frames:
            local_nodes = Railway.select_local_nodes(self.railway_map.railway_nodes, frame, r_ahead, r_behind)
            for node in local_nodes:
                if node not in nodes:
                    nodes.append(node)
        return nodes


    def __get_tracks_of_nodes__(self, nodes: list[TrackNode]):
        """
        Get tracks associated with each node.
        Args:
            nodes: List of TrackNode objects.
        Returns:
            Tuple (tracks, tracks_of_nodes).
        """
        tracks: list[TrackSegment] = []
        tracks_of_nodes: dict[TrackNode: list[TrackSegment]] = {}
        for node in nodes:
            tracks_of_node: list[TrackSegment] = []
            for track_id in self.railway_map.nodes_to_segment_assignment[node.id]:
                track = self.railway_map.railway_tracks[track_id]
                if track not in tracks:
                    tracks.append(track)
                tracks_of_node.append(track)     
            tracks_of_nodes[node] = tracks_of_node
        return tracks, tracks_of_nodes


    def __get_nodes_in_tracks__(self, tracks: list[TrackSegment], nodes: list[TrackNode]):
        """
        Get nodes associated with each track.
        Args:
            tracks: List of TrackSegment objects.
            nodes: List of TrackNode objects.
        Returns:
            Dictionary mapping TrackSegment to list of TrackNode.
        """
        nodes_in_tracks: dict[TrackSegment: list[TrackNode]] = {}
        for track in tracks:
            nodes_in_track: list[TrackNode] = []
            for node_id in self.railway_map.segment_to_node_assignment[track.id]:
                node = self.railway_map.railway_nodes[node_id]
                if node in nodes:
                    nodes_in_track.append(node)
            nodes_in_tracks[track] = nodes_in_track
        return nodes_in_tracks
    

    @staticmethod
    def __convert_nodes_to_gapless_2D_points_in_tracks__(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]], max_gap):
        """
        Interpolate nodes to create gapless 2D points along each track.
        Args:
            tracks: List of TrackSegment objects.
            nodes_in_tracks: Dictionary mapping TrackSegment to list of TrackNode.
            max_gap: Maximum allowed gap between points (meters).
        Returns:
            Dictionary mapping TrackSegment to list of 2D points (np.ndarray).
        """
        points_2D_in_tracks: dict[TrackSegment: list[np.ndarray]] = {}

        for track in tracks:

            nodes_in_track = nodes_in_tracks[track]
            points_2D_in_track = []

            previous_node = None
            previous_point_2D = None

            for node in nodes_in_track:
                point_2D = Railway.convert_node_to_point(node, False)

                if previous_node:
                    # if distance between nodes too large, add nodes in between
                    # linearly since large distance implies straight track
                    distance = np.linalg.norm(point_2D - previous_point_2D)
                    if distance > max_gap:
                        n = math.ceil(distance/max_gap)
                        # print("Distance between nodes too large", round(distance,1), ">", max_gap, " ... adding", n-1, "new points")
                        vector_2D = point_2D - previous_point_2D
                        for i in range(1, n):
                            new_point_2D = previous_point_2D + (i/n) * vector_2D
                            points_2D_in_track.append(new_point_2D)
                points_2D_in_track.append(point_2D)

                previous_node = node
                previous_point_2D = point_2D

                points_2D_in_tracks[track] = points_2D_in_track

        return points_2D_in_tracks
    

    @staticmethod
    def __convert_2D_to_3D_points_in_tracks__(tracks: list[TrackSegment], points_2D_in_tracks: dict[TrackSegment: list[np.ndarray]] = {}):
        """
        Add elevation to 2D points to create 3D points for each track.
        Args:
            tracks: List of TrackSegment objects.
            points_2D_in_tracks: Dictionary mapping TrackSegment to list of 2D points.
        Returns:
            Dictionary mapping TrackSegment to list of 3D points (np.ndarray).
        """
        points_3D_in_tracks: dict[TrackSegment: list[np.ndarray]] = {}

        for track in tracks:

            points_3D_in_track = []

            if not track.is_bridge:
                for point_2D in points_2D_in_tracks[track]:
                    point_3D = Railway.add_elevation_to_2D_point(point_2D)
                    points_3D_in_track.append(point_3D)

            elif track.is_bridge:
                points_2D = points_2D_in_tracks[track]

                first_point_3D = Railway.add_elevation_to_2D_point(points_2D[0])
                last_point_3D  = Railway.add_elevation_to_2D_point(points_2D[-1])

                points_3D_in_track.append(first_point_3D)

                vector_3D = last_point_3D - first_point_3D

                n = len(points_2D) - 2

                if n > 0:
                    for i, point_2D in enumerate(points_2D[1:-1]):

                        # For now, simple linear interpolation approach

                        new_point_3D = first_point_3D + ((i+1)/(n+1)) * vector_3D
                        points_3D_in_track.append(new_point_3D)
                    
                points_3D_in_track.append(last_point_3D)

            points_3D_in_tracks[track] = points_3D_in_track

        return points_3D_in_tracks
    

    @staticmethod
    def add_elevation_to_2D_point(point_2D: np.ndarray, custom_elevation: float=None) -> np.ndarray:
        """
        Add elevation to a 2D point to create a 3D point.

        Args:
            point_2D: 2D point as a numpy array.
            custom_elevation: Optional custom elevation value. If None, uses MapInfo.get_elevation.

        Returns:
            3D point as a numpy array.
        """
        assert point_2D.shape[0] == 2
        x = float(point_2D[0])
        y = float(point_2D[1])
        if custom_elevation != None:
            z = custom_elevation
        else:
            z = MapInfo.get_elevation(x, y)
        point_3D = np.array([[x], [y], [z]])
        assert point_3D.shape[0] == 3
        return point_3D


    @staticmethod
    def convert_node_to_point(node: TrackNode, add_elevation=True) -> np.ndarray:
        """
        Convert a TrackNode to a 2D or 3D point.

        Args:
            node: TrackNode object.
            add_elevation: If True, add elevation to create a 3D point.

        Returns:
            2D or 3D point as a numpy array.
        """
        x = node.x
        y = node.y
        if add_elevation:
            z = MapInfo.get_elevation(x, y)
            point = np.array([x, y, z])
        else:
            point = np.array([x, y])
        return point


    @staticmethod
    def convert_nodes_to_points(nodes: list[TrackNode], add_elevation=True) -> list[np.ndarray]:
        """
        Convert a list of TrackNode objects to 2D or 3D points.

        Args:
            nodes: List of TrackNode objects.
            add_elevation: If True, add elevation to create 3D points.

        Returns:
            List of 2D or 3D points as numpy arrays.
        """
        points = []
        for node in nodes:
            point = Railway.convert_node_to_point(node, add_elevation)
            points.append(point)
        return points


    @staticmethod
    def select_local_nodes(nodes: list[TrackNode], frame: 'Frame', r_ahead: float, r_behind: float = 0) -> list[TrackNode]:
        """
        Select railway nodes within a search radius ahead/behind a frame.

        Args:
            nodes: List of TrackNode objects.
            frame: Frame object.
            r_ahead: Search radius ahead (meters).
            r_behind: Search radius behind (meters).

        Returns:
            List of TrackNode objects within the specified radius.
        """
        heading, p_x, p_y = frame.gps.heading, frame.gps.x_w_gps, frame.gps.y_w_gps
        grad_x = np.cos(heading*np.pi/180)
        grad_y = np.sin(heading*np.pi/180)
        selected_nodes = []
        for node in nodes:
            x, y = node.x, node.y
            distance = ((x-p_x)**2 + (y-p_y)**2)**0.5
            dot_product = (x-p_x)*grad_x + (y-p_y)*grad_y
            cos_angle = dot_product / distance

            if distance < r_behind or (distance < r_ahead and cos_angle > 0.7):
                selected_nodes.append(node)

        return selected_nodes


    """
    Local Methods
    """

    def get_local_points_in_tracks(self, gps: 'GPS', r_ahead: float=None, r_behind: float=None, min_points: int=2):
        """
        Get local points in tracks for a given GPS position.

        Args:
            gps: GPS object.
            r_ahead: Optional search radius ahead (meters).
            r_behind: Optional search radius behind (meters).
            min_points: Minimum number of points per track (default: 2).

        Returns:
            Tuple (local_tracks, local_points_in_tracks):
                local_tracks: List of local TrackSegment objects.
                local_points_in_tracks: Dict mapping TrackSegment to list of points.
        """
        if r_ahead == None:
            r_ahead = self.r_ahead
        if r_behind == None:
            r_behind = self.r_behind

        local_tracks: list[TrackSegment] = []
        local_points_in_tracks: dict[TrackSegment: list[np.ndarray]] = {}

        heading, p_x, p_y = gps.heading, gps.x_w_gps, gps.y_w_gps
        grad_x = np.cos(heading*np.pi/180)
        grad_y = np.sin(heading*np.pi/180)

        for track in self.tracks:
            local_points_in_track = []
            for point in self.points_in_tracks_3D[track]:

                x, y = point[0], point[1]
                distance = ((x-p_x)**2 + (y-p_y)**2)**0.5
                dot_product = (x-p_x)*grad_x + (y-p_y)*grad_y
                cos_angle = dot_product / distance

                if distance < r_behind or (distance < r_ahead and cos_angle > 0.7):
                    local_points_in_track.append(point)
                    if track not in local_tracks:
                        local_tracks.append(track)
            
            if track in local_tracks:
                if len(local_points_in_track) < min_points:
                    local_tracks.remove(track)
                else:
                    local_points_in_tracks[track] = local_points_in_track

        return local_tracks, local_points_in_tracks


    # @staticmethod
    # def divide_into_continuous_segments(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):
    #     """ Output: list of list of nodes """
    #     segment_lists: list[list[TrackNode]] = []

    #     for track in tracks:
    #         nodes_in_track = nodes_in_tracks[track]

    #         split_index = None

    #         for i in range(2,len(nodes_in_track)-1):

    #             previous_node = nodes_in_track[i-1]
    #             current_node = nodes_in_track[i]

    #             if current_node.id not in railway_map.direct_neighbours_of_nodes[previous_node.id]:
    #                 if split_index: print("Error: multiple splits")
    #                 split_index = i
    #         if split_index:
    #             segment_lists.append(nodes_in_track[:split_index])
    #             segment_lists.append(nodes_in_track[split_index:])
    #         else:
    #             segment_lists.append(nodes_in_track)

    #     return segment_lists


    # @staticmethod
    # def find_connecting_nodes(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):
    #     """ Check if edge nodes of tracks are connecting nodes between multiple tracks. """

    #     edge_nodes: list[TrackNode] = []
    #     connecting_nodes: list[TrackNode] = []

    #     for track in tracks:
    #         nodes_in_track = nodes_in_tracks[track]
    #         for edge_node in [nodes_in_track[0], nodes_in_track[-1]]:
    #             if edge_node in edge_nodes:
    #                 connecting_nodes.append(edge_node)
    #             edge_nodes.append(edge_node)
    #     return connecting_nodes