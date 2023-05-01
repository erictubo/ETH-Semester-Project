#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External libraries
import plotly
import numpy as np

# Data & methods
from data import railway_map
from map_info import MapInfo
from visualisation import Visualisation
from transformation import Transformation

# Object types
from import_osm import (railway_map as RailwayMap, track_node as TrackNode, track_segment as TrackSegment)
from keyframe import KeyFrame


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

class RailwayPoint:

    def __init__(self, x, y, id=None):
        """ x,y in UTM coordinates [m] """
        self.x : float = x
        self.y : float = y
        # ID: only if node is original
        self.id : int = id
        self.z = MapInfo.get_elevation(x, y)
        self.segments : list[RailwaySegment] = []


class RailwaySegment:

    def __init__(self, id):
        self.id = id
        self.original_points : list[RailwayPoint] = []
        self.interpolated_points : list[RailwayPoint] = []





class Railway:
    """
    Data: railway nodes & tracks relevant to railway accross specified keyframes \\
    Methods:
    - ...
    """

    def __init__(self, keyframes: list[KeyFrame]):

        self.points, self.segments = self.__get_relevant_points_and_segments__(keyframes)

        self.interpolated_points, self.segments = self.__get_interpolated_points__(self.segments)

        # self.nodes = self.__get_relevant_nodes__(keyframes)
        # self.tracks, self.tracks_of_nodes = self.__get_tracks_of_nodes__(self.nodes)
        # self.nodes_in_tracks = self.__get_nodes_in_tracks__(self.tracks, self.nodes)

        # self.n_nodes = len(self.nodes)
        # self.n_tracks = len(self.tracks)

        # self.interpolated_points, self.interpolated_points_in_tracks = self.__get_interpolated_points__(self.tracks, self.nodes_in_tracks)

        # change existing lists to add elevation data        


    @staticmethod
    def plot_map():
        railway_map.plotly(plotly.graph_objs.Figure())


    @staticmethod
    def __get_relevant_points_and_segments__(keyframes: list[KeyFrame], r_ahead=80, r_behind=40):
        nodes: list[TrackNode] = []
        points: list[RailwayPoint] = []
        for keyframe in keyframes:
            local_nodes = Railway.select_local_nodes(railway_map.railway_nodes, keyframe, r_ahead, r_behind)
            for node in local_nodes:
                if node not in nodes:
                    nodes.append(node)
                    point = RailwayPoint(node.x, node.y, node.id)
                    points.append(point)

        tracks: list[TrackSegment] = []
        segments: list[RailwaySegment] = []
        for node in nodes:
            for track_id in railway_map.nodes_to_segment_assignment[node.id]:
                track = railway_map.railway_tracks[track_id]
                segment = RailwaySegment(id=track_id)
                if track not in tracks:
                    tracks.append(track) 
                    segments.append(segment)
        
        for segment, track in zip(segments, tracks):
            for point, node in zip(points, nodes):
                if track.id in railway_map.nodes_to_segment_assignment[node.id]:
                    segment.original_points.append(point)
                    point.segments.append(segment)

        return points, segments

    
    @staticmethod
    def __get_interpolated_points__(segments: list[RailwaySegment]):
        """ 2D map-space interpolation of tracks into segments """

        all_interpolated_points: list[RailwayPoint] = []

        for segment in segments:
            coordinates = []
            for point in segment.original_points:
                coordinate = np.array([[point.x], [point.y]])
                coordinates.append(coordinate)

            interpolated_coordinates = Transformation.interpolate_spline(coordinates, desired_spacing=20)

            interpolated_points = []
            for coordinate in interpolated_coordinates:
                x, y = float(coordinate[0]), float(coordinate[1])
                point = RailwayPoint(x, y)
                point.segments = [segment]
                interpolated_points.append(point)
                all_interpolated_points.append(point)

            segment.interpolated_points = interpolated_points

        return all_interpolated_points, segments
    


    # Better to interpolate in 3D directly ???


    # ToDo: connect consecutive tracks


    # ToDo: add elevation to all nodes -- needs to be added later to get accurate slopes

    # ToDo: add exception for bridge segments


    @staticmethod
    def convert_nodes_to_points(nodes: list[TrackNode], add_elevation=False) -> list[np.ndarray]:
        world_coordinates = []
        for node in nodes:
            x = node.x
            y = node.y
            if add_elevation:
                z = MapInfo.get_elevation(x, y)
                point = np.array([[x], [y], [z]])
            else:
                point = np.array([[x], [y]])
            world_coordinates.append(point)
        return world_coordinates



    @staticmethod
    def select_local_nodes(nodes: list[TrackNode], keyframe: KeyFrame, r_ahead: float, r_behind: float = 0):
        """ Selects local nodes at keyframe GPS pose, according to r_ahead, r_behind """
        heading, p_x, p_y = keyframe.GPS.heading, keyframe.GPS.x_w_gps, keyframe.GPS.y_w_gps
        grad_x = np.cos(heading*np.pi/180)
        grad_y = np.sin(heading*np.pi/180)
        selected_nodes: list[TrackNode] = []
        for node in nodes:
            dist2 = ((node.x-p_x)**2 + (node.y-p_y)**2)
            if dist2 < r_behind**2:
                selected_nodes.append(node)
            elif dist2 < r_ahead**2:
                if ((node.x - p_x) * grad_x + (node.y - p_y) * grad_y) > 0:
                    selected_nodes.append(node) 
        return selected_nodes
    

    
    

    """
    Advanced Methods
    """

    @staticmethod
    def find_connecting_nodes(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):
        """ Check if edge nodes of tracks are connecting nodes between multiple tracks. """

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
        """ Output: list of list of nodes """
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
    

    # @staticmethod
    # def add_neighbours_to_connecting_nodes(segment: list[TrackNode], num_neigbours=1):

    #     add neighbours to connecting nodes to improve splines
    #     num_neighbours depends on type of spline

    #     use transition segments (smallest opposite angle) to decide which node to add


    #     # add neighbours of start & end nodes
    #     # (points not in image but used for drawing lines)
    #     start_node, end_node = nodes_list[0], nodes_list[-1]
    #     for node in [start_node, end_node]:
            
    #         # find neighbour_in_nodes_list
    #         for neighbour_index in railway_map.direct_neighbours_of_nodes[node.id]:
    #             neighbour = railway_map.railway_nodes[neighbour_index]
    #             if neighbour in nodes_list:
    #                 neighbour_in_nodes_list = neighbour

    #         for neighbour_index in railway_map.direct_neighbours_of_nodes[node.id]:
    #             neighbour = railway_map.railway_nodes[neighbour_index]
    #             if neighbour not in nodes_list:
    #                 # check that neighbour is in opposite direction to neighbour_in_list
    #                 # (independent of track, for interpolation)
    #                 # i.e. dot product negative
    #                 if ((neighbour.x-node.x) * (neighbour_in_nodes_list.x-node.x) + \
    #                 (neighbour.y-node.y) * (neighbour_in_nodes_list.y-node.y)) < 0:
    #                     if node == start_node: nodes_list.insert(0, neighbour)
    #                     elif node == end_node: nodes_list.append(neighbour)

    #     return segment
    

        # @staticmethod
        # def get_railway_nodes_by_tracks(keyframe: KeyFrame, r_ahead: float, r_behind: float = 0):

        #     tracks: list[TrackSegment] = []
        #     nodes_in_tracks: dict[TrackSegment: list[TrackNode]] = {}

        #     for track in railway_map.railway_tracks:
        #         nodes_in_track: list[TrackNode] = []

        #         # Get all nodes in track
        #         for node_id in railway_map.segment_to_node_assignment[track.id]:
        #             node = railway_map.railway_nodes[node_id]
        #             nodes_in_track.append(node)
                
        #         # Select nodes according to filters
        #         nodes_in_track = MapInfo.select_local_railway_nodes(nodes_in_track, p_x, p_y, heading, r_ahead, r_behind)

        #         if len(nodes_in_track) > 0:
        #             tracks.append(track)
        #             nodes_in_tracks[track] = nodes_in_track

        #     return tracks, nodes_in_tracks
        

        # @staticmethod
        # def switch_to_railway_tracks_by_nodes(tracks: list[TrackSegment], nodes_in_tracks: dict[TrackSegment: list[TrackNode]]):

        #     """
        #     Output: dictionary with nodes and their tracks, list of nodes that are in multiple tracks
        #     """
        #     nodes: list[TrackNode] = []
        #     tracks_of_nodes: dict[TrackNode: list[TrackSegment]] = []
        #     nodes_in_multiple_tracks: list[TrackNode] = []

        #     for track in tracks:
        #         nodes_in_track = nodes_in_tracks[track]

        #         for node in nodes_in_track:

        #             if node not in nodes:
        #                 nodes.append(node)

        #             track_ids_of_node = railway_map.nodes_to_segment_assignment[node.id]

        #             # If node is in multiple tracks
        #             if len(track_ids_of_node) > 1:
        #                 tracks_of_node = []
        #                 # Fill dictionary with all tracks of the node
        #                 for track_id_of_node in track_ids_of_node:
        #                     track_of_node = railway_map.railway_tracks[track_id_of_node]
        #                     tracks_of_node.append(track_of_node)
        #                 # Append node to dictionary
        #                 nodes_in_multiple_tracks.append(node)

        #             else:
        #                 tracks_of_node = track
        #             tracks_of_nodes[node] = tracks_of_node

        #     return nodes, tracks_of_nodes