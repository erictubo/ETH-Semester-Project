#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External libraries
import plotly
import numpy as np
import math
import matplotlib.pyplot as plt

# Data & methods
from data import path_to_osm_file
from map_info import MapInfo
from visualisation import Visualisation
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
    Processed railway in combined map area of given keyframes
    - this will import relevant railway nodes, interpolate gaps between railway nodes,
    add elevation data, create compatible tracks, etc.
    - trade-off: requires initial pre-processing time, but enables much faster reprojection loop
    """

    def __init__(self, frames: list['Frame'], max_gap: float, r_ahead: float, r_behind: float):

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
        self.railway_map.plotly(plotly.graph_objs.Figure())

    
    def visualise_2D(self, show_tracks=True, show_nodes=True, frames: list['Frame'] = []):
        if show_tracks:
            for track in self.tracks:
                for point in self.points_in_tracks_2D[track]:
                    Visualisation.plot_XY(point[0], point[1], 'orange')
        if show_nodes:
            for node in self.nodes:
                Visualisation.plot_XY(node.x, node.y, color='red')
        if len(frames) > 0:
            for frame in frames:
                point = frame.gps.t_w_gps
                Visualisation.plot_XY(point[0], point[1], 'black')
                #plt.Circle((point[0], point[1]), r_ahead, 'red')
        plt.xlabel("Position X [m]")
        plt.ylabel("Position Y [m]")
        Visualisation.show_plot()

    def visualise_3D(self, frames: list['Frame'] = []):
        ax = Visualisation.create_3D_plot("All points in tracks")
        for track in self.tracks:
            color = 'blue'
            if track.is_bridge:
                color = 'cyan'
            points = self.points_in_tracks_3D[track]
            Visualisation.plot_3D_points(ax, points, color)
        for frame in frames:
            point = frame.gps.t_w_gps
            Visualisation.plot_3D_points(ax, [point], 'red')
        Visualisation.show_plot()


    # Hidden methods: called at initialisation

    def __get_relevant_nodes__(self, frames: list['Frame'], r_ahead: float, r_behind: float):
        nodes: list[TrackNode] = []
        for frame in frames:
            local_nodes = Railway.select_local_nodes(self.railway_map.railway_nodes, frame, r_ahead, r_behind)
            for node in local_nodes:
                if node not in nodes:
                    nodes.append(node)
        return nodes


    def __get_tracks_of_nodes__(self, nodes: list[TrackNode]):
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
    def add_elevation_to_2D_point(point_2D: np.ndarray, custom_elevation: float=None):

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
        points = []
        for node in nodes:
            Railway.convert_node_to_point(node, add_elevation)
        return points


    @staticmethod
    def select_local_nodes(nodes: list[TrackNode], frame: 'Frame', r_ahead: float, r_behind: float = 0):
        """ Selects local nodes at frame GPS pose, according to r_ahead, r_behind """
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