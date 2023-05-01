import xml.etree.ElementTree as ET
import utm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.path as mplPath
from matplotlib.colors import ListedColormap
# from ekf import ExtendedPathConstrainedKF
import plotly.graph_objects as go
from itertools import combinations
import re
from tqdm import tqdm
from collections import namedtuple
from math import sqrt



class track_segment:
    def __init__(self, id, osm_id):
        """
        Initializes a track segment
        :param id: track segment id
        :type id: int
        :param osm_id: osm id
        :type osm_id: str
        """
        self.id = id                        # Track ID (also index in list of tracks)
        self.osm_id = osm_id                # OSM Track ID (not needed)
        self.track_node_ids = []            # ordered list of nodes along track
        self.x_cors = []                    # Coordinates of nodes
        self.y_cors = []
        self.arc_length = []                # Integrated arclength along track
        self.curvature = []                 # not used
        self.max_speed = 0                  # Speedlimit from OSM
        self.type = ''                      # not used,
        self.num_tracks = 1                 # Issue with OSM map, where multiple tracks are represented as one. Nothing I can do
        self.is_transition = False          # Transition segment or not
        self.transitions = []               # Transition Regions

        self.is_bridge = False


    def print(self):
        print("---")
        print(("ID: ").ljust(27, ' ') + str(self.id))
        print(("OSM-ID: ").ljust(27, ' ') + self.osm_id)
        print(("Type: ").ljust(27, ' ') + self.type)
        print(("Speed: ").ljust(27, ' ') + str(self.max_speed))
        print(("Number of Tracks: ").ljust(27, ' ') + str(self.num_tracks))
        print(("Type: ").ljust(27, ' ') + self.type)
        print(("Nodes: ").ljust(27, ' ') + str(self.track_node_ids))
        print(("X: ").ljust(27, ' ') + str(self.x_cors))
        print(("Y: ").ljust(27, ' ') + str(self.y_cors))

    def plot_segment(self,axis,color):
        axis.plot(self.x_cors,self.y_cors,c=color)
        axis.scatter(self.x_cors[0],self.y_cors[0],c=color)
        axis.scatter(self.x_cors[-1],self.y_cors[-1],c=color)

# Node that defines tracks
class track_node:
    def __init__(self,id,osm_id):
        self.id = id                # ID and index
        self.osm_id = osm_id        # OSM ID, not used
        self.lat = 0.0
        self.lon = 0.0
        self.x = 0.0                #UTM coordinates
        self.y = 0.0
        self.zone = ''

    def print(self):
        print("---")
        print(("ID: ").ljust(27, ' ') + str(self.id))
        print(("OSM-ID: ").ljust(27, ' ') + str(self.osm_id))
        print(("Lat: ").ljust(27, ' ') + self.lat)
        print(("Lon: ").ljust(27, ' ') + self.lon)
        print(("X: ").ljust(27, ' ') + str(self.x))
        print(("Y: ").ljust(27, ' ') + str(self.y))
        print(("Zone: ").ljust(27, ' ') + self.zone)
        # print(("Nodes: ").ljust(27, ' ') + str(self.nodes))

    def assign_lat_long(self,lat,lon):
        # Convert lat,lon to UTM
        self.lat = lat
        self.lon = lon
        u = utm.from_latlon(float(lat), float(lon))
        self.x = u[0]
        self.y = u[1]
        self.zone = str(u[2]) + u[3]

# Whole Map object
class railway_map:
    def __init__(self,name):
        self.name = name                                # Name
        self.file = ""                                  # OSM Filename
        self.osm_track_ids = []                         # List of osm track ids (not used)
        self.osm_node_ids = []                          # List of osm node ids (not used)
        self.nodes_to_segment_assignment = []           # Assignment list of which nodes belong to which tracks
        self.segment_to_node_assignment = []            # Assignment of which tracks connect to a node
        self.direct_neighbours_of_nodes = []            # Which nodes are directly connected to each other
        self.edge_to_edge_connectivits = []             # Which tracks are directly connected to each other
        self.railway_tracks = []                        # Acutal list of tracks
        self.railway_nodes = []                         # Actual list of nodes
        self.transition_length = 0                      # Length of a transition segment (actual half length, length per track that is connected)

    def print(self):
        print(self.name, " - " , self.file)

    # Plotting OSM Railway data
    def plotly(self,figure):
        # Plot with plotly
        end_points_x = []
        end_points_y = []
        map_fig = go.Figure()
        # Get Zone
        zone_num = re.findall(r'\d+', self.railway_nodes[0].zone)[0]
        zone_letter = re.findall(r'[a-zA-Z]', self.railway_nodes[0].zone)[0]

        for way in self.railway_tracks:
            if not way.is_transition:
                end_points_x.append(way.x_cors[0])
                end_points_x.append(way.x_cors[-1])
                end_points_y.append(way.y_cors[0])
                end_points_y.append(way.y_cors[-1])
                lat = []
                lon = []
                for i in range(len(way.x_cors)):
                    x_p = way.x_cors[i]
                    y_p = way.y_cors[i]
                    res = utm.to_latlon(x_p, y_p, int(zone_num), zone_letter)
                    lat.append(res[0])
                    lon.append(res[1])
                map_fig.add_trace(go.Scattermapbox(
                    mode = "lines",
                    lon = lon,
                    lat = lat,
                    marker = {'size': 10},
                    line=dict(color='blue', width=2)))

                figure.add_trace(go.Scatter(x=way.x_cors, y=way.y_cors, name='Segment '+str(way.id),mode='lines',line=dict(color='royalblue', width=2),showlegend=False))

                # figure.add_trace(go.Scatter(x=way.x_cors, y=way.y_cors, name='Segment '+str(way.id),mode='lines',line=dict(color='royalblue', width=2),showlegend=True))
                # figure.add_trace(go.Scatter(x=way.x_cors, y=way.y_cors, name='Segment '+str(way.id),mode='lines',line=dict(color='royalblue', width=2),showlegend=False,line_shape='spline'))
        figure.add_trace(go.Scatter(x=end_points_x, y=end_points_y, name='Segment-End-Points',mode='markers',line=dict(color='royalblue', width=2),showlegend=True))
        map_fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[end_points_x],
            lat=[end_points_y],
            marker={'size': 10},
            line=dict(color='red', width=2)))

        map_fig.update_layout(
            margin ={'l':0,'t':0,'b':0,'r':0},
            mapbox = {
                # 'center': {'lon': 10, 'lat': 10},
                'style': "stamen-terrain",
                'center': {'lat': 47.367, 'lon': 8.54},
                'zoom': 10})

        map_fig.show()



    # Generate transition segments
    def generate_transitions(self, trans_len, figure=None):
        """
        Generate transition segments
        :param trans_len: transition segment length
        :type trans_len: int
        :param figure: plot if figure is insterted
        :type figure: plotly figure
        """
        self.transition_length = trans_len
        transition_length = self.transition_length   # Length of transition space on each track, in meters

        # Find all transition points
        transition_point_ids = [ x for x in range(len(self.nodes_to_segment_assignment)) if len(self.nodes_to_segment_assignment[x]) > 1]
        # print(transition_point_ids)

        transition_point_x = [self.railway_nodes[x].x for x in transition_point_ids]
        transition_point_y = [self.railway_nodes[x].y for x in transition_point_ids]

        if figure is not None:
            # Plot transition points
            figure.add_trace(go.Scatter(x=transition_point_x, y=transition_point_y, name='Transition Points',mode='markers',line=dict(color='red', width=2),showlegend=True,visible = "legendonly"))

        # Determine which segments to create
        segments_to_generate = []
        for transition_point in transition_point_ids:
            if transition_point == 2635:
                h = 1+2
            # Get all possible combinations of transitions at an intersection point
            comb = combinations(self.nodes_to_segment_assignment[transition_point], 2)
            # print("---")
            # print(self.nodes_to_segment_assignment[transition_point])
            for com in comb:
                segment = [transition_point,com[0],com[1]]
                # print(segment)
                segments_to_generate.append(segment)
                # print(com[1])
        # print(len(segments_to_generate))

        # Create transition segments
        generated_segments = []
        for trans_seg in segments_to_generate:

            # id of point, origin track, target track
            point_id = trans_seg[0]
            track_1_id = trans_seg[1]
            track_2_id = trans_seg[2]
            if track_1_id == 910 or track_2_id == 910:
                h = 1+1
            # Point on Track 1
            x_0 = []
            y_0 = []
            arc_0 = []
            # Transition Point
            x_1 = self.railway_nodes[point_id].x
            y_1 = self.railway_nodes[point_id].y
            center_point_idx = self.railway_tracks[track_1_id].track_node_ids.index(point_id)
            arc_1_1 = self.railway_tracks[track_1_id].arc_length[center_point_idx]
            center_point_idx = self.railway_tracks[track_2_id].track_node_ids.index(point_id)
            arc_1_2 = self.railway_tracks[track_2_id].arc_length[center_point_idx]
            # Point on Track 2
            x_2 = []
            y_2 = []
            arc_2 = []
            # 3 possibilities, the point can be at the beginning, middle or end of each segment...

            # Find the next point on track 1 that connects to the transition point. needed to define transition segment shape
            # First point
            if point_id == self.railway_tracks[track_1_id].track_node_ids[0]:
                # print("Beginning Point")
                track_position = 0 + transition_length
                x_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                arc_0.append(track_position)
            if point_id == self.railway_tracks[track_1_id].track_node_ids[-1]:
                # print("End Point")
                track_position = self.railway_tracks[track_1_id].arc_length[-1] - transition_length
                x_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                arc_0.append(track_position)
            if point_id != self.railway_tracks[track_1_id].track_node_ids[0] and point_id != self.railway_tracks[track_1_id].track_node_ids[-1]:
                # print("Center Point, I will deal with this later")
                arc_len_sub = arc_1_1 - transition_length
                arc_len_add = arc_1_1 + transition_length
                arc_0.append(arc_len_sub)
                arc_0.append(arc_len_add)
                x_0.append(np.interp(arc_len_sub, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(arc_len_sub, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                x_0.append(np.interp(arc_len_add, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(arc_len_add, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))

            # Find the next point on track 1 that connects to the transition point. needed to define transition segment shape
            # Second Point
            if point_id == self.railway_tracks[track_2_id].track_node_ids[0]:
                # print("Beginning Point")
                track_position = 0 + transition_length
                x_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                arc_2.append(track_position)
            if point_id == self.railway_tracks[track_2_id].track_node_ids[-1]:
                # print("End Point")
                track_position = self.railway_tracks[track_2_id].arc_length[-1] - transition_length
                x_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                arc_2.append(track_position)
            if point_id != self.railway_tracks[track_2_id].track_node_ids[0] and point_id != self.railway_tracks[track_2_id].track_node_ids[-1]:
                # print("Center Point, I will deal with this later")
                arc_len_sub = arc_1_2 - transition_length
                arc_len_add = arc_1_2 + transition_length
                arc_2.append(arc_len_sub)
                arc_2.append(arc_len_add)
                x_2.append(np.interp(arc_len_sub, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(arc_len_sub, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                x_2.append(np.interp(arc_len_add, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(arc_len_add, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))

            #  Find all combinations
            for i in range(len(x_0)):
                for j in range(len(x_2)):
                    # Check angle acceptable, feasable for a train:
                    is_acceptable, angle = is_angle_acceptable([x_1,y_1], [x_0[i],y_0[i]], [x_2[j],y_2[j]])

                    if is_acceptable:
                        x_points = [x_0[i],x_1,x_2[j]]
                        y_points = [y_0[i],y_1,y_2[j]]

                        transition_1 = [np.min([arc_0[i],arc_1_1]),np.max([arc_0[i],arc_1_1]),track_1_id]
                        transition_2 = [np.min([arc_2[j],arc_1_2]),np.max([arc_2[j],arc_1_2]),track_2_id]
                        generated_segments.append([x_points,y_points,transition_1,transition_2,angle])
        # print(generated_segments)

        # Clear Edge-to-Edge Connectivity
        # Because we are working with transition segments, we don't need the normal connectivity graph
        # Instead, each connection is to a transition, and from there to the next edge, so we have to redo the graph
        self.edge_to_edge_connectivits.clear()
        total_num_tracks = len(self.railway_tracks) + len(generated_segments)
        # Generate empty edges
        for i in range(total_num_tracks):
            self.edge_to_edge_connectivits.append([])

        plot_transition_x = []
        plot_transition_y = []
        for segment in generated_segments:
            # print(segment)
            new_id = len(self.railway_tracks)
            # print("--")
            # print(new_id)
            # print(segment[4])
            for i in segment[0]:
                plot_transition_x.append(i)
            for i in segment[1]:
                plot_transition_y.append(i)
            plot_transition_x.append(np.nan)
            plot_transition_y.append(np.nan)

            if figure is not None:
                figure.add_trace(go.Scatter(x=segment[0], y=segment[1], name='Trans ' + str(new_id),mode='lines',line=dict(color='red', width=2)))
            # Create Segment
            new_transition_track = track_segment(new_id,"0")
            new_transition_track.is_transition = True
            # # new_transition_track.track_node_ids = []
            new_transition_track.x_cors = segment[0]
            new_transition_track.y_cors = segment[1]
            new_transition_track.arc_length = [0,transition_length,2*transition_length]
            # # new_transition_track.curvature = []
            new_transition_track.transitions.append([0,transition_length,segment[2][2]])
            new_transition_track.transitions.append([transition_length,2*transition_length,segment[3][2]])
            # print(new_transition_track.transitions)
            self.railway_tracks.append(new_transition_track)

            # Add Transition Data to existing track
            self.railway_tracks[segment[2][2]].transitions.append([segment[2][0],segment[2][1],new_id])
            self.railway_tracks[segment[3][2]].transitions.append([segment[3][0],segment[3][1],new_id])

            # Fillind edge_to_edge connectivity graph
            self.edge_to_edge_connectivits[new_id].append(segment[2][2])
            self.edge_to_edge_connectivits[new_id].append(segment[3][2])
            self.edge_to_edge_connectivits[segment[2][2]].append(new_id)
            self.edge_to_edge_connectivits[segment[3][2]].append(new_id)

        if figure is not None:
            figure.add_trace(go.Scatter(x=plot_transition_x, y=plot_transition_y, name='Transitions ',mode='lines',line=dict(color='red', width=2),visible = "legendonly"))


    def circles_from_p1p2r(self, p1, p2, r):
        """
        Calculate arc for curved transition segments
        :param p1: point 1
        :type p1: np.array
        :param p2: point2
        :type p2: np.array
        :param r: radius
        :type r: float
        :return:
        :rtype:
        """

        Pt = namedtuple('Pt', 'x, y')
        Circle = Cir = namedtuple('Circle', 'x, y, r')
        'Following explanation at http://mathforum.org/library/drmath/view/53027.html'
        if r == 0.0:
            raise ValueError('radius of zero')
        (x1, y1), (x2, y2) = p1, p2
        if p1 == p2:
            raise ValueError('coincident points gives infinite number of Circles')
        # delta x, delta y between points
        dx, dy = x2 - x1, y2 - y1
        # dist between points
        q = sqrt(dx ** 2 + dy ** 2)
        if q > 2.0 * r:
            raise ValueError('separation of points > diameter')
        # halfway point
        x3, y3 = (x1 + x2) / 2, (y1 + y2) / 2
        # distance along the mirror line
        d = sqrt(r ** 2 - (q / 2) ** 2)
        # One answer
        c1 = Cir(x=x3 - d * dy / q,
                 y=y3 + d * dx / q,
                 r=abs(r))
        # The other answer
        c2 = Cir(x=x3 + d * dy / q,
                 y=y3 - d * dx / q,
                 r=abs(r))
        return c1, c2

    def circle_func(self, x_points, radius, center):
        y_new_1 = [-np.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1] for x in x_points]
        y_new_2 = [np.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1] for x in x_points]
        return y_new_1, y_new_2

    def find_candidate(self, x_points, node_point, circles):
        distances = []
        for circle in range(len(circles)):
            d = 0
            for point in range(len(x_points)):
                d += np.sqrt((x_points[point] - node_point[0]) ** 2 + (circles[circle][point] - node_point[1]) ** 2)

            distances.append(d)
        min_arg = np.argmin(distances)
        return min_arg

    def find_arc_points(self, point1, point2, node_point, radius=1.0, number_points=5):
        x_min = np.min([point1[0], point2[0]])
        x_max = np.max([point1[0], point2[0]])
        c1, c2 = self.circles_from_p1p2r(point1, point2, radius)
        x_new = np.linspace(x_min, x_max, number_points)
        y_new_c1_1, y_new_c1_2 = self.circle_func(x_new.copy(), c1.r, (c1.x, c1.y))
        y_new_c2_1, y_new_c2_2 = self.circle_func(x_new.copy(), c2.r, (c2.x, c2.y))
        circle_array = [y_new_c1_1, y_new_c1_2, y_new_c2_1, y_new_c2_2]
        index = self.find_candidate(x_new, node_point, circle_array)
        return x_new, circle_array[index]

    def distance_points(self, point1, point2, smooth=0, floor=True):
        dist = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        if floor is True:
            if dist < 1:
                dist = dist + smooth
            else:
                dist = np.floor(dist) + smooth
        return dist

    def generate_transitions_vehicle_bent(self,trans_len, figure=None):
        """
        Generate arc transition segments for vehicles
        :param trans_len: trans segment length
        :type trans_len: int
        :param figure: plot if figure is inserted
        :type figure: plotly figure
        """
        self.transition_length = trans_len
        # How long should straight segment be before curve
        self.transition_straight = 3
        self.transition_length_curve_start = self.transition_length - self.transition_straight
        # Find all transition points
        transition_point_ids = [ x for x in range(len(self.nodes_to_segment_assignment)) if len(self.nodes_to_segment_assignment[x]) > 1]
        # print(transition_point_ids)
        transition_point_x = [self.railway_nodes[x].x for x in transition_point_ids]
        transition_point_y = [self.railway_nodes[x].y for x in transition_point_ids]
        if figure is not None:
        # Plot transition points
            figure.add_trace(go.Scatter(x=transition_point_x, y=transition_point_y, name='Transition Points',mode='markers',line=dict(color='red', width=2),showlegend=True,visible = "legendonly"))

        # Determine which segments to create
        segments_to_generate = []
        for transition_point in transition_point_ids:

            # Get all possible combinations of transitions at an intersection point
            comb = combinations(self.nodes_to_segment_assignment[transition_point], 2)
            # print("---")
            # print(self.nodes_to_segment_assignment[transition_point])
            for com in comb:

                segment = [transition_point,com[0],com[1]]
                # print(segment)
                segments_to_generate.append(segment)
                # print(com[1])
        # print(len(segments_to_generate))

        # Create transition segments
        generated_segments = []
        for trans_seg in segments_to_generate:

            # id of point, origin track, target track
            point_id = trans_seg[0]
            track_1_id = trans_seg[1]
            track_2_id = trans_seg[2]

            # Point on Track 1
            x_0 = []
            y_0 = []
            arc_0 = []
            # Arc start on Track 1
            x_0_a = []
            y_0_a = []
            arc_0_a = []
            # Transition Point
            x_1 = self.railway_nodes[point_id].x
            y_1 = self.railway_nodes[point_id].y
            arc_1_1_is_center = False
            center_point_idx = self.railway_tracks[track_1_id].track_node_ids.index(point_id)
            arc_1_1 = self.railway_tracks[track_1_id].arc_length[center_point_idx] #+ self.transition_length_curve_start
            if arc_1_1 == 0:
                arc_1_1 = arc_1_1 + self.transition_length_curve_start

            elif arc_1_1 > 0:
                if center_point_idx == len(self.railway_tracks[track_1_id].arc_length) - 1:
                    arc_1_1 = arc_1_1 - self.transition_length_curve_start
                else:
                    arc_1_1_is_center = True

            arc_1_2_is_center = False
            center_point_idx = self.railway_tracks[track_2_id].track_node_ids.index(point_id)
            arc_1_2 = self.railway_tracks[track_2_id].arc_length[center_point_idx] #- self.transition_length_curve_start
            if arc_1_2 == 0:
                #arc_1_2 = arc_1_2 + self.transition_length * (2/3)
                arc_1_2 = arc_1_2 + self.transition_length_curve_start
            elif arc_1_2 > 0:
                if center_point_idx == len(self.railway_tracks[track_2_id].arc_length) - 1:
                    arc_1_2 = arc_1_2 - self.transition_length_curve_start
                else:
                    arc_1_2_is_center = True
            # Point on Track 2
            x_2 = []
            y_2 = []
            arc_2 = []
            # Arc start on Track 2
            x_2_a = []
            y_2_a = []
            arc_2_a = []
            # 3 possibilities, the point can be at the beginning, middle or end of each segment...

            # Find the next point on track 1 that connects to the transition point. needed to define transition segment shape
            # First point
            if point_id == self.railway_tracks[track_1_id].track_node_ids[0]:
                # print("Beginning Point")
                #track_position = 0 + transition_length
                track_position = self.transition_length
                #track_position_arc_start = self.transition_length * (2/3)#self.transition_length_curve_start
                track_position_arc_start = self.transition_length_curve_start  # self.transition_length_curve_start

                x_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                arc_0.append(track_position)

                x_0_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                arc_0_a.append(track_position_arc_start)

            elif point_id == self.railway_tracks[track_1_id].track_node_ids[-1]:
                # print("End Point")
                #track_position = self.railway_tracks[track_1_id].arc_length[-1] - transition_length
                track_position = self.railway_tracks[track_1_id].arc_length[-1] - self.transition_length
                track_position_arc_start = self.railway_tracks[track_1_id].arc_length[-1] - self.transition_length_curve_start

                x_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(track_position, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                arc_0.append(track_position)

                x_0_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                arc_0_a.append(track_position_arc_start)
            else:
                # print("Center Point, I will deal with this later")
                arc_len_sub = arc_1_1 - self.transition_length
                arc_len_add = arc_1_1 + self.transition_length
                arc_0.append(arc_len_sub)
                arc_0.append(arc_len_add)

                #arc_len_sub_a = arc_1_1 - self.transition_length*(2/3)
                arc_len_sub_a = arc_1_1 - self.transition_length_curve_start
                #arc_len_add_a = arc_1_1 + self.transition_length*(2/3)
                arc_len_add_a = arc_1_1 + self.transition_length_curve_start
                arc_0_a.append(arc_len_sub_a)
                arc_0_a.append(arc_len_add_a)

                x_0.append(np.interp(arc_len_sub, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(arc_len_sub, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                x_0.append(np.interp(arc_len_add, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0.append(np.interp(arc_len_add, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))

                x_0_a.append(np.interp(arc_len_sub_a, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0_a.append(np.interp(arc_len_sub_a, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))
                x_0_a.append(np.interp(arc_len_add_a, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].x_cors))
                y_0_a.append(np.interp(arc_len_add_a, self.railway_tracks[track_1_id].arc_length, self.railway_tracks[track_1_id].y_cors))

            # Find the next point on track 1 that connects to the transition point. needed to define transition segment shape
            # Second Point
            if point_id == self.railway_tracks[track_2_id].track_node_ids[0]:
                # print("Beginning Point")
                track_position = self.transition_length
                track_position_arc_start = self.transition_length_curve_start  # self.transition_length_curve_start

                x_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                arc_2.append(track_position)

                x_2_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_2_id].arc_length,
                                     self.railway_tracks[track_2_id].x_cors))
                y_2_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_2_id].arc_length,
                                     self.railway_tracks[track_2_id].y_cors))
                arc_2_a.append(track_position_arc_start)

            elif point_id == self.railway_tracks[track_2_id].track_node_ids[-1]:
                track_position = self.railway_tracks[track_2_id].arc_length[-1] - self.transition_length
                track_position_arc_start = self.railway_tracks[track_2_id].arc_length[
                                               -1] - self.transition_length_curve_start
                x_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(track_position, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                arc_2.append(track_position)

                x_2_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2_a.append(np.interp(track_position_arc_start, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                arc_2_a.append(track_position_arc_start)

            else:
                arc_len_sub = arc_1_2 - self.transition_length
                arc_len_add = arc_1_2 + self.transition_length
                arc_len_sub_a = arc_1_2 - self.transition_length_curve_start
                arc_len_add_a = arc_1_2 + self.transition_length_curve_start
                arc_2.append(arc_len_sub)
                arc_2.append(arc_len_add)
                x_2.append(np.interp(arc_len_sub, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(arc_len_sub, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                x_2.append(np.interp(arc_len_add, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2.append(np.interp(arc_len_add, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                arc_2_a.append(arc_len_sub_a)
                arc_2_a.append(arc_len_add_a)
                x_2_a.append(np.interp(arc_len_sub_a, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2_a.append(np.interp(arc_len_sub_a, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))
                x_2_a.append(np.interp(arc_len_add_a, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].x_cors))
                y_2_a.append(np.interp(arc_len_add_a, self.railway_tracks[track_2_id].arc_length, self.railway_tracks[track_2_id].y_cors))

            #  Find all combinations
            for i in range(len(x_0)):
                for j in range(len(x_2)):
                    # Check angle acceptable, feasable for a train:
                    is_acceptable, angle = is_angle_acceptable([x_1,y_1], [x_0[i],y_0[i]], [x_2[j],y_2[j]])
                    is_acceptable =True

                    if is_acceptable:

                        x_points = [x_0[i],x_1,x_2[j]]
                        y_points = [y_0[i],y_1,y_2[j]]
                        x_points_arc = [x_0_a[i], x_1, x_2_a[j]]
                        y_points_arc = [y_0_a[i], y_1, y_2_a[j]]
                        if arc_1_1_is_center is True:
                            if i == 0:
                                transition_1 = [np.min([arc_0[i],arc_1_1 - self.transition_length_curve_start ]),np.max([arc_0[i],arc_1_1 - self.transition_length_curve_start]),track_1_id]
                            else:
                                transition_1 = [np.min([arc_0[i],arc_1_1 + - self.transition_length_curve_start]),np.max([arc_0[i],arc_1_1 + - self.transition_length_curve_start]),track_1_id]
                        else:
                            transition_1 = [np.min([arc_0[i],arc_1_1]),np.max([arc_0[i],arc_1_1]),track_1_id]

                        if arc_1_2_is_center is True:
                            if j == 0:
                                transition_2 = [np.min([arc_2[j],arc_1_2 - self.transition_length_curve_start]),np.max([arc_2[j],arc_1_2 - self.transition_length_curve_start]),track_2_id]
                            else:
                                transition_2 = [np.min([arc_2[j], arc_1_2 + self.transition_length_curve_start]),
                                                np.max([arc_2[j], arc_1_2 + self.transition_length_curve_start]),
                                                track_2_id]
                        else:
                            transition_2 = [np.min([arc_2[j],arc_1_2]),np.max([arc_2[j],arc_1_2]),track_2_id]
                        generated_segments.append([x_points,y_points,transition_1,transition_2,angle, x_points_arc, y_points_arc])
        # print(generated_segments)

        # Clear Edge-to-Edge Connectivity
        # Because we are working with transition segments, we don't need the normal connectivity graph
        # Instead, each connection is to a transition, and from there to the next edge, so we have to redo the graph
        self.edge_to_edge_connectivits.clear()
        total_num_tracks = len(self.railway_tracks) + len(generated_segments)
        # Generate empty edges
        for i in range(total_num_tracks):
            self.edge_to_edge_connectivits.append([])

        plot_transition_x = []
        plot_transition_y = []
        for segment in generated_segments:
            # print(segment)
            new_id = len(self.railway_tracks)

            point1 = [segment[0][0], segment[1][0]]
            point2 = [segment[0][2], segment[1][2]]
            point1_arc = [segment[5][0], segment[6][0]]
            point2_arc = [segment[5][2], segment[6][2]]

            middle_node = [segment[0][1], segment[1][1]]
            angle = segment[4]

            if point1_arc != point2_arc and angle < 130 and angle > 50:
                try:
                    circle_points_x, circle_points_y = self.find_arc_points(point1_arc, point2_arc, middle_node,
                                         radius=self.distance_points(point1_arc, point2_arc, 1.5),
                                         number_points=10)
                except:
                    h = 1+1

                whole_segment_x = circle_points_x
                whole_segment_y = circle_points_y
                first_circle_point = [circle_points_x[0], circle_points_y[0]]
                last_circle_point = [circle_points_x[-1], circle_points_y[-1]]

                segment_distance_to_circle_arc1 = [self.distance_points(point1, first_circle_point, floor=False), self.distance_points(point1, last_circle_point, floor=False)]
                point_closest = np.argmin(segment_distance_to_circle_arc1)

                if point_closest == 0:
                    whole_segment_x = np.insert(circle_points_x, 0, point1[0])
                    whole_segment_x = np.insert(whole_segment_x, whole_segment_x.shape[0], point2[0])
                    whole_segment_y = np.insert(circle_points_y, 0, point1[1])
                    whole_segment_y = np.insert(whole_segment_y, whole_segment_y.shape[0], point2[1])

                else:
                    whole_segment_x = np.insert(circle_points_x, 0, point2[0])
                    whole_segment_x = np.insert(whole_segment_x, whole_segment_x.shape[0], point1[0])
                    whole_segment_y = np.insert(circle_points_y, 0, point2[1])
                    whole_segment_y = np.insert(whole_segment_y, whole_segment_y.shape[0], point1[1])



            else:

                whole_segment_x = np.array([point1[0], point1_arc[0], point2_arc[0], point2[0]])
                whole_segment_y = np.array([point1[1], point1_arc[1], point2_arc[1], point2[1]])


                plot_transition_x.append(point1[0])
                plot_transition_x.append(point2[0])

                plot_transition_y.append(point1[1])
                plot_transition_y.append(point2[1])

                plot_transition_x.append(np.nan)
                plot_transition_y.append(np.nan)


            if figure is not None:
                figure.add_trace(go.Scatter(x=whole_segment_x, y=whole_segment_y, name='Trans ' + str(new_id),mode='lines',line=dict(color='red', width=2)))
            # Create Segment
            new_transition_track = track_segment(new_id,"0")
            new_transition_track.is_transition = True
            # # new_transition_track.track_node_ids = []


            new_transition_track.x_cors = whole_segment_x#[segment[0][0], segment[0][2]]
            new_transition_track.y_cors = whole_segment_y#[segment[1][0], segment[1][2]]
            length = [0]
            total_length = 0

            for i in range(len(whole_segment_x) - 1):
                total_length_temp = np.sqrt((whole_segment_x[i + 1] - whole_segment_x[i])**2 + (whole_segment_y[i+1] - whole_segment_y[i])**2)
                length.append(total_length + total_length_temp)
                total_length = total_length + total_length_temp


                # new_transition_track.x_cors = whole_segment_x[::-1]  # [segment[0][0], segment[0][2]]
                # new_transition_track.y_cors = whole_segment_y[::-1]  # [segment[1][0], segment[1][2]]
                # length = length[::-1]

            new_transition_track.arc_length = length#[0, 2*transition_length]#[0,transition_length,2*transition_length]
            # # new_transition_track.curvature = []
            #new_transition_track.transitions.append([self.transition_length_curve_start,self.transition_length,segment[2][2]])
            if point1_arc[0] > point2_arc[0]:
                new_transition_track.transitions.append([0, self.transition_length_curve_start, segment[3][2]])
                new_transition_track.transitions.append([length[-1] - self.transition_length_curve_start,length[-1],segment[2][2]])
            else:
                new_transition_track.transitions.append([0, self.transition_length_curve_start, segment[2][2]])
                new_transition_track.transitions.append(
                    [length[-1] - self.transition_length_curve_start, length[-1], segment[3][2]])
            # print(new_transition_track.transitions)
            self.railway_tracks.append(new_transition_track)

            # Add Transition Data to existing track
            add_info = self.railway_tracks[segment[2][2]]
            add_transitions = self.railway_tracks[segment[2][2]].transitions
            self.railway_tracks[segment[2][2]].transitions.append([segment[2][0],segment[2][1],new_id])
            self.railway_tracks[segment[3][2]].transitions.append([segment[3][0],segment[3][1],new_id])

            # Fillind edge_to_edge connectivity graph
            self.edge_to_edge_connectivits[new_id].append(segment[2][2])
            self.edge_to_edge_connectivits[new_id].append(segment[3][2])
            self.edge_to_edge_connectivits[segment[2][2]].append(new_id)
            self.edge_to_edge_connectivits[segment[3][2]].append(new_id)


        #figure.add_trace(go.Scatter(x=plot_transition_x, y=plot_transition_y, name='Transitions ',mode='lines',line=dict(color='red', width=2),visible = "legendonly"))


    # Importing OSM file for cars
    def import_from_osm_file_vehicles(self,file_name):
        """
        Importing OSM file for cars
        :param file_name: map filename
        :type file_name: str
        """

        self.file = file_name
        print("Loading OSM data...")
        # railway_types_to_ignore = ['disused','abandoned','construction', 'platform', 'rail']

        # Ignore elements with these keys
        # Ignore 'rail' when dealing with tram, and vice versa
        railway_types_to_ignore = ['pedestrian', 'service', 'track', 'bus_guideway', 'escape', 'raceway', 'busway',
                                   'tram', 'rail', 'disused','abandoned','construction', 'tram_stop', 'preserved',
                                   'station', 'subway_entrance', 'halt', 'platform','miniature','turntable',
                                   'traverser', 'monorail', 'narrow_gauge', 'crossing', 'level_crossing',
                                   'buffer_stop', 'signal', 'proposed', 'footway', 'bridleway', 'steps', 'corridor',
                                   'sidewalk', 'crossing', 'cycleway', 'share_busway', 'opposite_share_busway',
                                   'unclassified' , 'residential']


        need = ["primary", "secondary", "tertiary", "residential"]

        railway_types_to_ignore = ['pedestrian', 'service', 'track', 'bus_guideway', 'escape', 'raceway', 'busway',
                                   'tram', 'rail', 'disused','abandoned','construction', 'tram_stop', 'preserved',
                                   'station', 'subway_entrance', 'halt', 'platform','miniature','turntable',
                                   'traverser', 'monorail', 'narrow_gauge', 'crossing', 'level_crossing',
                                   'buffer_stop', 'signal', 'proposed', 'footway', 'bridleway', 'steps', 'corridor',
                                   'sidewalk', 'crossing', 'cycleway', 'share_busway', 'opposite_share_busway',
                                   'trunk', 'motorway', 'unclassified', 'living_street',
                                   'road', 'footway', 'bridleway', 'steps', 'corridor', 'path'
                                   ]

        # Parse xml tree
        tree = ET.parse(self.file)
        print("File parsed...")

        root = tree.getroot()

        # Do first parsing pass to find all tracks
        for way in tqdm(root.findall('way')):
            tag_node_list = [x.tag for x in way]
            for node in way:
                if (node.tag != 'nd') and (node.tag != 'tag'):
                    print("Some unexpected nodes showed up:", node.tag)
            is_railway = False
            max_speed = 0
            type = ''
            num_tracks = 1
            tagslist = [x.attrib['k'] for x in way.findall('tag')]
            for node in way.findall('tag'):
                # These are Attributes
                if node.attrib.get('k') == 'highway':
                    # print('This is a railway')
                    is_railway = True
                    type = node.attrib.get('v')
                if node.attrib['k'] == 'maxspeed':

                    max_speed = node.attrib.get('v')
                    try:
                        max_speed = int(max_speed)
                    except:
                        pass

                if node.attrib['k'] == 'tracks':
                    try:
                        num_tracks = int(node.attrib.get('v'))
                    except:
                        pass
                    # print('Tracks: ',num_tracks)
                    if num_tracks > 1:
                        print("WARNING!! INSUFFICITENT TRACK DATA (multiple tracks in one way)")

            if is_railway and not type in railway_types_to_ignore:
                # Storing this as a track in osm_track_id_database
                osm_way_id = way.attrib.get('id')
                # If it doeasn't exist yes, create it
                if not osm_way_id in self.osm_track_ids:
                    self.osm_track_ids.append(osm_way_id)
                    self.segment_to_node_assignment.append([])
                    self.edge_to_edge_connectivits.append([])
                    way_index = self.osm_track_ids.index(osm_way_id)    # assign an index
                    new_track = track_segment(way_index,osm_way_id)     # create track object
                    new_track.max_speed = max_speed
                    new_track.type = type
                    new_track.num_tracks = num_tracks
                    self.railway_tracks.append(new_track)               # add to map object
                    # print(way_id)
                else:
                    print("Saw this track before: ", osm_way_id)
                way_index = self.osm_track_ids.index(osm_way_id)

                # These are Way-Points that need to be parsed. they will be added to the track data later after parsing
                for node in way.findall('nd'):
                    osm_node_id = node.attrib.get('ref')
                    # If node is new
                    if not osm_node_id in self.osm_node_ids:
                        # print("New Node: ", node_id)
                        self.osm_node_ids.append(osm_node_id)
                        node_index = self.osm_node_ids.index(osm_node_id)           # Assign node index
                        self.nodes_to_segment_assignment.append([])                 # Create placeholdes in tables
                        self.direct_neighbours_of_nodes.append([])
                        new_node_entry = track_node(node_index,osm_node_id)         # create node
                        self.railway_nodes.append(new_node_entry)                   # Add to map
                        self.segment_to_node_assignment[way_index].append(node_index)   # Fill tables
                        self.nodes_to_segment_assignment[node_index].append(way_index)
                    else:
                        node_index = self.osm_node_ids.index(osm_node_id)
                        self.segment_to_node_assignment[way_index].append(node_index)
                        self.nodes_to_segment_assignment[node_index].append(way_index)
        # Import all the node Geometry

        print("Found tracks...")
        for node in tqdm(root.findall('node')):
            osm_id = node.attrib['id']
            if osm_id in self.osm_node_ids:
                lat = node.attrib['lat']
                lon = node.attrib['lon']
                node_index = self.osm_node_ids.index(osm_id)
                if node_index != self.railway_nodes[node_index].id:
                    print("NODE ID ALERT")
                if osm_id != self.railway_nodes[node_index].osm_id:
                    print("NODE OSM-ID ALERT")
                self.railway_nodes[node_index].assign_lat_long(lat,lon) # assigning geographic points to each node

        # Now add track geometry to tracks:
        for way in tqdm(root.findall('way')):
            osm_way_id = way.attrib['id']
            if osm_way_id in self.osm_track_ids:
                way_index = self.osm_track_ids.index(osm_way_id)
                list_of_contained_osm_node_ids = []
                for node in way.findall('nd'):
                    list_of_contained_osm_node_ids.append(node.attrib['ref']) # get all node ids for this track

                for i in range(len(list_of_contained_osm_node_ids)):
                    node_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i]) # get current node

                    if not node_index in self.segment_to_node_assignment[way_index]:
                        print("This node doesn't belong here?")
                    self.railway_tracks[way_index].track_node_ids.append(node_index)            # Assign node to track
                    self.railway_tracks[way_index].x_cors.append(self.railway_nodes[node_index].x)
                    self.railway_tracks[way_index].y_cors.append(self.railway_nodes[node_index].y)

                    # Set Edge Connectivity DATA
                    for reachable_segment in self.nodes_to_segment_assignment[node_index]:
                        if reachable_segment != way_index:
                            if not reachable_segment in self.edge_to_edge_connectivits[way_index]:
                                self.edge_to_edge_connectivits[way_index].append(reachable_segment)
                                # print("....")
                                # print(way_index)
                                # print(reachable_segment)
                    # Set node Neighborhood Data
                    if i > 0:
                        neighbor_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i-1])
                        if not neighbor_index in self.direct_neighbours_of_nodes[node_index]:
                            self.direct_neighbours_of_nodes[node_index].append(neighbor_index)
                    if i < len(list_of_contained_osm_node_ids)-1:
                        neighbor_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i+1])
                        if not neighbor_index in self.direct_neighbours_of_nodes[node_index]:
                            self.direct_neighbours_of_nodes[node_index].append(neighbor_index)


        # Computing Track ArcLength
        for way in tqdm(self.railway_tracks):
            # print(way.id)
            way.arc_length.append(0.0)
            for i in range(1,len(way.track_node_ids)):
                distance = np.sqrt((way.x_cors[i]-way.x_cors[i-1])**2 + (way.y_cors[i]-way.y_cors[i-1])**2)
                new_total = way.arc_length[-1] + distance
                way.arc_length.append(new_total)


    # Importing OSM file
    def import_from_osm_file(self,file_name):
        self.file = file_name
        print("Loading OSM data...")
        # railway_types_to_ignore = ['disused','abandoned','construction', 'platform', 'rail']

        # Ignore elements with these keys
        # Ignore 'rail' when dealing with tram, and vice versa
        railway_types_to_ignore = ['rail', 'disused','abandoned','construction', 'preserved', 'station', 'subway_entrance', 'halt', 'platform','miniature','turntable','traverser', 'monorail', 'narrow_gauge', 'crossing', 'level_crossing', 'buffer_stop', 'signal', 'proposed']
        # Parse xml tree
        tree = ET.parse(self.file)
        print("File parsed...")

        root = tree.getroot()

        # Do first parsing pass to find all tracks
        for way in root.findall('way'):
            for node in way:
                if (node.tag != 'nd') and (node.tag != 'tag'):
                    print("Some unexpected nodes showed up:", node.tag)
            is_railway = False
            is_bridge = False
            max_speed = 0
            type = ''
            num_tracks = 1
            for node in way.findall('tag'):
                # These are Attributes
                if node.attrib['k'] == 'railway':
                    # print('This is a railway')
                    is_railway = True
                    type = node.attrib['v']
                if node.attrib['k'] == 'maxspeed':
                    max_speed = int(node.attrib['v'])
                if node.attrib['k'] == 'tracks':
                    num_tracks = int(node.attrib['v'])
                    # print('Tracks: ',num_tracks)
                    if num_tracks > 1:
                        print("WARNING!! INSUFFICITENT TRACK DATA (multiple tracks in one way)")

                if node.attrib['k'] == 'bridge':
                    is_bridge = True

            if is_railway and not type in railway_types_to_ignore:
                # Storing this as a track in osm_track_id_database
                osm_way_id = way.attrib['id']
                # If it doeasn't exist yes, create it
                if not osm_way_id in self.osm_track_ids:
                    self.osm_track_ids.append(osm_way_id)
                    self.segment_to_node_assignment.append([])
                    self.edge_to_edge_connectivits.append([])
                    way_index = self.osm_track_ids.index(osm_way_id)    # assign an index
                    new_track = track_segment(way_index,osm_way_id)     # create track object
                    new_track.is_bridge = is_bridge
                    new_track.max_speed = max_speed
                    new_track.type = type
                    new_track.num_tracks = num_tracks
                    self.railway_tracks.append(new_track)               # add to map object
                    # print(way_id)
                else:
                    print("Saw this track before: ", osm_way_id)
                way_index = self.osm_track_ids.index(osm_way_id)

                # These are Way-Points that need to be parsed. they will be added to the track data later after parsing
                for node in way.findall('nd'):
                    osm_node_id = node.attrib['ref']
                    # If node is new
                    if not osm_node_id in self.osm_node_ids:
                        # print("New Node: ", node_id)
                        self.osm_node_ids.append(osm_node_id)
                        node_index = self.osm_node_ids.index(osm_node_id)           # Assign node index
                        self.nodes_to_segment_assignment.append([])                 # Create placeholdes in tables
                        self.direct_neighbours_of_nodes.append([])
                        new_node_entry = track_node(node_index,osm_node_id)         # create node
                        self.railway_nodes.append(new_node_entry)                   # Add to map
                        self.segment_to_node_assignment[way_index].append(node_index)   # Fill tables
                        self.nodes_to_segment_assignment[node_index].append(way_index)
                    else:
                        node_index = self.osm_node_ids.index(osm_node_id)
                        self.segment_to_node_assignment[way_index].append(node_index)
                        self.nodes_to_segment_assignment[node_index].append(way_index)
        # Import all the node Geometry

        print("Found tracks...")
        for node in root.findall('node'):
            osm_id = node.attrib['id']
            if osm_id in self.osm_node_ids:
                lat = node.attrib['lat']
                lon = node.attrib['lon']
                node_index = self.osm_node_ids.index(osm_id)
                if node_index != self.railway_nodes[node_index].id:
                    print("NODE ID ALERT")
                if osm_id != self.railway_nodes[node_index].osm_id:
                    print("NODE OSM-ID ALERT")
                self.railway_nodes[node_index].assign_lat_long(lat,lon) # assigning geographic points to each node

        # Now add track geometry to tracks:
        for way in root.findall('way'):
            osm_way_id = way.attrib['id']
            if osm_way_id in self.osm_track_ids:
                way_index = self.osm_track_ids.index(osm_way_id)
                list_of_contained_osm_node_ids = []
                for node in way.findall('nd'):
                    list_of_contained_osm_node_ids.append(node.attrib['ref']) # get all node ids for this track

                for i in range(len(list_of_contained_osm_node_ids)):
                    node_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i]) # get current node

                    if not node_index in self.segment_to_node_assignment[way_index]:
                        print("This node doesn't belong here?")
                    self.railway_tracks[way_index].track_node_ids.append(node_index)            # Assign node to track
                    self.railway_tracks[way_index].x_cors.append(self.railway_nodes[node_index].x)
                    self.railway_tracks[way_index].y_cors.append(self.railway_nodes[node_index].y)

                    # Set Edge Connectivity DATA
                    for reachable_segment in self.nodes_to_segment_assignment[node_index]:
                        if reachable_segment != way_index:
                            if not reachable_segment in self.edge_to_edge_connectivits[way_index]:
                                self.edge_to_edge_connectivits[way_index].append(reachable_segment)
                                # print("....")
                                # print(way_index)
                                # print(reachable_segment)
                    # Set node Neighborhood Data
                    if i > 0:
                        neighbor_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i-1])
                        if not neighbor_index in self.direct_neighbours_of_nodes[node_index]:
                            self.direct_neighbours_of_nodes[node_index].append(neighbor_index)
                    if i < len(list_of_contained_osm_node_ids)-1:
                        neighbor_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i+1])
                        if not neighbor_index in self.direct_neighbours_of_nodes[node_index]:
                            self.direct_neighbours_of_nodes[node_index].append(neighbor_index)

        # Computing Track Curvatures (This is actually never used!!!)
        for way in self.railway_tracks:
            # print(way.id)
            for i in range(len(way.track_node_ids)):
                left_node = np.array([0,0])
                center_node = np.array([0,0])
                right_node = np.array([0,0])
                if i == 0:
                    # print("Find prev")
                    center_node = np.array([way.x_cors[i],way.y_cors[i]])
                    right_node = np.array([way.x_cors[i+1],way.y_cors[i+1]])
                    # left_node = np.array([way.x_cors[i-1],way.y_cors[i-1]])
                    neighbors = self.direct_neighbours_of_nodes[way.track_node_ids[i]].copy()
                    # print(neighbors)
                    neighbors.remove(way.track_node_ids[i+1])
                    if (len(neighbors) == 1):
                        # Only one neighboring node
                        left_node = np.array([self.railway_nodes[neighbors[0]].x,self.railway_nodes[neighbors[0]].y])
                    elif (len(neighbors) == 0):
                        left_node = center_node
                    else:
                        # Multiple neigboring nodes, need to check them:
                        angles = []
                        for neighbor in neighbors:
                            is_acceptable, angle = is_angle_acceptable([way.x_cors[i],way.y_cors[i]], [way.x_cors[i+1],way.y_cors[i+1]], [self.railway_nodes[neighbor].x,self.railway_nodes[neighbor].y])
                            angles.append(angle)
                            # print(neighbors)
                            # print(angles)
                        best_one = neighbors[angles.index(min(angles))]
                        left_node = np.array([self.railway_nodes[best_one].x,self.railway_nodes[best_one].y])


                elif i == len(way.track_node_ids) - 1:
                    # print("find seq")
                    left_node = np.array([way.x_cors[i-1],way.y_cors[i-1]])
                    center_node = np.array([way.x_cors[i],way.y_cors[i]])
                    # right_node = np.array([way.x_cors[i+1],way.y_cors[i+1]])

                    neighbors = self.direct_neighbours_of_nodes[way.track_node_ids[i]].copy()
                    # print(neighbors)
                    neighbors.remove(way.track_node_ids[i-1])
                    if (len(neighbors) == 1):
                        # Only one neighboring node
                        right_node = np.array([self.railway_nodes[neighbors[0]].x,self.railway_nodes[neighbors[0]].y])
                    elif (len(neighbors) == 0):
                        right_node = center_node
                    else:
                        # Multiple neigboring nodes, need to check them:
                        angles = []
                        for neighbor in neighbors:
                            is_acceptable, angle = is_angle_acceptable([way.x_cors[i],way.y_cors[i]], [way.x_cors[i-1],way.y_cors[i-1]], [self.railway_nodes[neighbor].x,self.railway_nodes[neighbor].y])
                            angles.append(angle)
                            # print(neighbors)
                            # print(angles)
                        best_one = neighbors[angles.index(min(angles))]
                        right_node = np.array([self.railway_nodes[best_one].x,self.railway_nodes[best_one].y])


                else:
                    left_node = np.array([way.x_cors[i-1],way.y_cors[i-1]])
                    center_node = np.array([way.x_cors[i],way.y_cors[i]])
                    right_node = np.array([way.x_cors[i+1],way.y_cors[i+1]])
                curvature = compute_curvature(left_node,center_node,right_node)
                self.railway_tracks[way.id].curvature.append(curvature)
            # print(self.railway_tracks[way.id].curvature)
            # print(way.track_node_ids)

        # Computing Track ArcLength
        for way in self.railway_tracks:
            # print(way.id)
            way.arc_length.append(0.0)
            for i in range(1,len(way.track_node_ids)):
                distance = np.sqrt((way.x_cors[i]-way.x_cors[i-1])**2 + (way.y_cors[i]-way.y_cors[i-1])**2)
                new_total = way.arc_length[-1] + distance
                way.arc_length.append(new_total)


    def import_from_osm_file_vehicles(self,file_name):
        self.file = file_name
        print("Loading OSM data...")
        # railway_types_to_ignore = ['disused','abandoned','construction', 'platform', 'rail']

        # Ignore elements with these keys
        # Ignore 'rail' when dealing with tram, and vice versa
        railway_types_to_ignore = ['pedestrian', 'service', 'track', 'bus_guideway', 'escape', 'raceway', 'busway',
                                   'tram', 'rail', 'disused','abandoned','construction', 'tram_stop', 'preserved',
                                   'station', 'subway_entrance', 'halt', 'platform','miniature','turntable',
                                   'traverser', 'monorail', 'narrow_gauge', 'crossing', 'level_crossing',
                                   'buffer_stop', 'signal', 'proposed', 'footway', 'bridleway', 'steps', 'corridor',
                                   'sidewalk', 'crossing', 'cycleway', 'share_busway', 'opposite_share_busway',
                                   'unclassified' , 'residential']


        need = ["primary", "secondary", "tertiary", "residential"]

        railway_types_to_ignore = ['pedestrian', 'service', 'track', 'bus_guideway', 'escape', 'raceway', 'busway',
                                   'tram', 'rail', 'disused','abandoned','construction', 'tram_stop', 'preserved',
                                   'station', 'subway_entrance', 'halt', 'platform','miniature','turntable',
                                   'traverser', 'monorail', 'narrow_gauge', 'crossing', 'level_crossing',
                                   'buffer_stop', 'signal', 'proposed', 'footway', 'bridleway', 'steps', 'corridor',
                                   'sidewalk', 'crossing', 'cycleway', 'share_busway', 'opposite_share_busway',
                                   'trunk', 'motorway', 'unclassified', 'living_street',
                                   'road', 'footway', 'bridleway', 'steps', 'corridor', 'path'
                                   ]

        # Parse xml tree
        tree = ET.parse(self.file)
        print("File parsed...")

        root = tree.getroot()

        # Do first parsing pass to find all tracks
        for way in tqdm(root.findall('way')):
            tag_node_list = [x.tag for x in way]
            for node in way:
                if (node.tag != 'nd') and (node.tag != 'tag'):
                    print("Some unexpected nodes showed up:", node.tag)
            is_railway = False
            max_speed = 0
            type = ''
            num_tracks = 1
            tagslist = [x.attrib['k'] for x in way.findall('tag')]
            for node in way.findall('tag'):
                # These are Attributes
                if node.attrib.get('k') == 'highway':
                    # print('This is a railway')
                    is_railway = True
                    type = node.attrib.get('v')
                if node.attrib['k'] == 'maxspeed':

                    max_speed = node.attrib.get('v')
                    try:
                        max_speed = int(max_speed)
                    except:
                        pass

                if node.attrib['k'] == 'tracks':
                    try:
                        num_tracks = int(node.attrib.get('v'))
                    except:
                        pass
                    # print('Tracks: ',num_tracks)
                    if num_tracks > 1:
                        print("WARNING!! INSUFFICITENT TRACK DATA (multiple tracks in one way)")

            if is_railway and not type in railway_types_to_ignore:
                # Storing this as a track in osm_track_id_database
                osm_way_id = way.attrib.get('id')
                # If it doeasn't exist yes, create it
                if not osm_way_id in self.osm_track_ids:
                    self.osm_track_ids.append(osm_way_id)
                    self.segment_to_node_assignment.append([])
                    self.edge_to_edge_connectivits.append([])
                    way_index = self.osm_track_ids.index(osm_way_id)    # assign an index
                    new_track = track_segment(way_index,osm_way_id)     # create track object
                    new_track.max_speed = max_speed
                    new_track.type = type
                    new_track.num_tracks = num_tracks
                    self.railway_tracks.append(new_track)               # add to map object
                    # print(way_id)
                else:
                    print("Saw this track before: ", osm_way_id)
                way_index = self.osm_track_ids.index(osm_way_id)

                # These are Way-Points that need to be parsed. they will be added to the track data later after parsing
                for node in way.findall('nd'):
                    osm_node_id = node.attrib.get('ref')
                    # If node is new
                    if not osm_node_id in self.osm_node_ids:
                        # print("New Node: ", node_id)
                        self.osm_node_ids.append(osm_node_id)
                        node_index = self.osm_node_ids.index(osm_node_id)           # Assign node index
                        self.nodes_to_segment_assignment.append([])                 # Create placeholdes in tables
                        self.direct_neighbours_of_nodes.append([])
                        new_node_entry = track_node(node_index,osm_node_id)         # create node
                        self.railway_nodes.append(new_node_entry)                   # Add to map
                        self.segment_to_node_assignment[way_index].append(node_index)   # Fill tables
                        self.nodes_to_segment_assignment[node_index].append(way_index)
                    else:
                        node_index = self.osm_node_ids.index(osm_node_id)
                        self.segment_to_node_assignment[way_index].append(node_index)
                        self.nodes_to_segment_assignment[node_index].append(way_index)
        # Import all the node Geometry

        print("Found tracks...")
        for node in tqdm(root.findall('node')):
            osm_id = node.attrib['id']
            if osm_id in self.osm_node_ids:
                lat = node.attrib['lat']
                lon = node.attrib['lon']
                node_index = self.osm_node_ids.index(osm_id)
                if node_index != self.railway_nodes[node_index].id:
                    print("NODE ID ALERT")
                if osm_id != self.railway_nodes[node_index].osm_id:
                    print("NODE OSM-ID ALERT")
                self.railway_nodes[node_index].assign_lat_long(lat,lon) # assigning geographic points to each node

        # Now add track geometry to tracks:
        for way in tqdm(root.findall('way')):
            osm_way_id = way.attrib['id']
            if osm_way_id in self.osm_track_ids:
                way_index = self.osm_track_ids.index(osm_way_id)
                list_of_contained_osm_node_ids = []
                for node in way.findall('nd'):
                    list_of_contained_osm_node_ids.append(node.attrib['ref']) # get all node ids for this track

                for i in range(len(list_of_contained_osm_node_ids)):
                    node_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i]) # get current node

                    if not node_index in self.segment_to_node_assignment[way_index]:
                        print("This node doesn't belong here?")
                    self.railway_tracks[way_index].track_node_ids.append(node_index)            # Assign node to track
                    self.railway_tracks[way_index].x_cors.append(self.railway_nodes[node_index].x)
                    self.railway_tracks[way_index].y_cors.append(self.railway_nodes[node_index].y)

                    # Set Edge Connectivity DATA
                    for reachable_segment in self.nodes_to_segment_assignment[node_index]:
                        if reachable_segment != way_index:
                            if not reachable_segment in self.edge_to_edge_connectivits[way_index]:
                                self.edge_to_edge_connectivits[way_index].append(reachable_segment)
                                # print("....")
                                # print(way_index)
                                # print(reachable_segment)
                    # Set node Neighborhood Data
                    if i > 0:
                        neighbor_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i-1])
                        if not neighbor_index in self.direct_neighbours_of_nodes[node_index]:
                            self.direct_neighbours_of_nodes[node_index].append(neighbor_index)
                    if i < len(list_of_contained_osm_node_ids)-1:
                        neighbor_index = self.osm_node_ids.index(list_of_contained_osm_node_ids[i+1])
                        if not neighbor_index in self.direct_neighbours_of_nodes[node_index]:
                            self.direct_neighbours_of_nodes[node_index].append(neighbor_index)


        # Computing Track ArcLength
        for way in tqdm(self.railway_tracks):
            # print(way.id)
            way.arc_length.append(0.0)
            for i in range(1,len(way.track_node_ids)):
                distance = np.sqrt((way.x_cors[i]-way.x_cors[i-1])**2 + (way.y_cors[i]-way.y_cors[i-1])**2)
                new_total = way.arc_length[-1] + distance
                way.arc_length.append(new_total)



# Generating random path
def create_random_path(map, length):
    number_network_nodes = len(map.osm_node_ids)
    random_id = int(np.floor(number_network_nodes * np.random.rand(1))) # random start point
    # axis.scatter(map.railway_nodes[random_id].x,map.railway_nodes[random_id].y,c='r')
    point_ids = [random_id]
    x_points = [map.railway_nodes[random_id].x]
    y_points = [map.railway_nodes[random_id].y]
    length_of_path = 0
    accumulated_length = [0]

    # Get allowed speed
    segment_id = map.nodes_to_segment_assignment[point_ids[0]][0]
    max_speed_at_node = map.railway_tracks[segment_id].max_speed
    # print(max_speed_at_node)
    if max_speed_at_node == 0:
        max_speed_at_node = 80
    max_speeds = [max_speed_at_node]

    # Collect ground-truth data
    true_ids = []
    true_s = []
    # for i in range(100):
    while(accumulated_length[-1] < length):
        #print(accumulated_length[-1])
        cur_id = point_ids[-1]
        # print(cur_id)
        feasable_ids = []
        # Find all connecting nodes
        for id in map.direct_neighbours_of_nodes[cur_id]:
            if id != cur_id:
                if len(point_ids) > 1:
                    if id != point_ids[-2]:
                        # find all connecting nodes that a train could actually reach
                        is_acceptable, angle = is_angle_acceptable([x_points[-1],y_points[-1]], [x_points[-2],y_points[-2]], [map.railway_nodes[id].x,map.railway_nodes[id].y])

                        if is_acceptable:
                            feasable_ids.append(id)
                else:
                    feasable_ids.append(id)
        # print(feasable_ids)
        if len(feasable_ids) < 1:
            # no reachable node
            print("Reached End of Map")
            break

        if len(feasable_ids) > 0:
            # Select random id:
            new_id = feasable_ids[int(np.floor(len(feasable_ids) * np.random.rand(1)))]
            # add point to path
            point_ids.append(new_id)
            x_points.append(map.railway_nodes[new_id].x)
            y_points.append(map.railway_nodes[new_id].y)
            # add distance travelled
            distance = np.sqrt((x_points[-1]-x_points[-2])**2 + (y_points[-1]-y_points[-2])**2)
            length_of_path += distance
            accumulated_length.append(length_of_path)
            segment_id = map.nodes_to_segment_assignment[new_id][0]
            # get id of connecting segment
            cur_segments = map.nodes_to_segment_assignment[cur_id]
            new_segments = map.nodes_to_segment_assignment[new_id]
            connecting_segments = []
            for old_seg in cur_segments:
                for new_seg in new_segments:
                    if old_seg == new_seg:
                        connecting_segments.append(old_seg)
            true_ids.append(connecting_segments[0])

            # get new speed limit
            max_speed_at_node = map.railway_tracks[segment_id].max_speed
            # print(max_speed_at_node)
            if max_speed_at_node == 0:
                max_speed_at_node = 80
            max_speeds.append(max_speed_at_node)
            # axis.scatter(x_points[-1],y_points[-1],c='r')
            # axis.annotate(point_ids[-1], (x_points[-1],y_points[-1]),fontsize=10)
    # print(len(accumulated_length))
    # print(len(true_ids))
    # print(true_ids)
    return x_points, y_points, max_speeds, accumulated_length, true_ids


# Generating random path
def create_random_path_vehicle(map, length):
    number_network_nodes = len(map.osm_node_ids)
    random_id = int(np.floor(number_network_nodes * np.random.rand(1))) # random start point
    # axis.scatter(map.railway_nodes[random_id].x,map.railway_nodes[random_id].y,c='r')
    point_ids = [random_id]
    x_points = [map.railway_nodes[random_id].x]
    y_points = [map.railway_nodes[random_id].y]
    length_of_path = 0
    accumulated_length = [0]

    # Get allowed speed
    segment_id = map.nodes_to_segment_assignment[point_ids[0]][0]
    max_speed_at_node = map.railway_tracks[segment_id].max_speed
    # print(max_speed_at_node)
    if max_speed_at_node == 0:
        max_speed_at_node = 80
    max_speeds = [max_speed_at_node]

    # Collect ground-truth data
    true_ids = []
    true_s = []
    # for i in range(100):
    while(accumulated_length[-1] < length):
        #print(accumulated_length[-1])
        cur_id = point_ids[-1]
        # print(cur_id)
        feasable_ids = []
        # Find all connecting nodes
        for id in map.direct_neighbours_of_nodes[cur_id]:
            if id != cur_id:
                if len(point_ids) > 1:
                    if id != point_ids[-2]:
                        # find all connecting nodes that a train could actually reach
                        is_acceptable, angle = is_angle_acceptable([x_points[-1],y_points[-1]], [x_points[-2],y_points[-2]], [map.railway_nodes[id].x,map.railway_nodes[id].y])
                        is_acceptable = True
                        if is_acceptable:
                            feasable_ids.append(id)
                else:
                    feasable_ids.append(id)
        # print(feasable_ids)
        if len(feasable_ids) < 1:
            # no reachable node
            print("Reached End of Map")
            break

        if len(feasable_ids) > 0:
            # Select random id:
            new_id = feasable_ids[int(np.floor(len(feasable_ids) * np.random.rand(1)))]
            # add point to path
            point_ids.append(new_id)
            x_points.append(map.railway_nodes[new_id].x)
            y_points.append(map.railway_nodes[new_id].y)
            # add distance travelled
            distance = np.sqrt((x_points[-1]-x_points[-2])**2 + (y_points[-1]-y_points[-2])**2)
            length_of_path += distance
            accumulated_length.append(length_of_path)
            segment_id = map.nodes_to_segment_assignment[new_id][0]
            # get id of connecting segment
            cur_segments = map.nodes_to_segment_assignment[cur_id]
            new_segments = map.nodes_to_segment_assignment[new_id]
            connecting_segments = []
            for old_seg in cur_segments:
                for new_seg in new_segments:
                    if old_seg == new_seg:
                        connecting_segments.append(old_seg)
            true_ids.append(connecting_segments[0])

            # get new speed limit
            max_speed_at_node = map.railway_tracks[segment_id].max_speed
            # print(max_speed_at_node)
            if max_speed_at_node == 0:
                max_speed_at_node = 80
            max_speeds.append(max_speed_at_node)
            # axis.scatter(x_points[-1],y_points[-1],c='r')
            # axis.annotate(point_ids[-1], (x_points[-1],y_points[-1]),fontsize=10)
    # print(len(accumulated_length))
    # print(len(true_ids))
    # print(true_ids)
    return x_points, y_points, max_speeds, accumulated_length, true_ids





def is_angle_acceptable(cur_point, last_point, next_point):
    # check angle spanned by 3 points to see if it is traversable for a train
    cur_point = np.asarray(cur_point)
    last_point = np.asarray(last_point)
    next_point = np.asarray(next_point)
    vector_last_to_now = cur_point - last_point
    vector_now_to_next = next_point - cur_point
    unit_vector_1 = vector_last_to_now / np.linalg.norm(vector_last_to_now)
    unit_vector_2 = vector_now_to_next / np.linalg.norm(vector_now_to_next)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    # print(dot_product)
    angle = np.arccos(dot_product) *180/np.pi
    # print(vector_last_to_now)
    # print(vector_now_to_next)
    # print(angle)
    # This assumes trains can make instantaneous 40 degree turns. Of course not realisitic, but good enough for now
    if angle < 40:
        return True, angle
    else:
        return False, angle


def find_close_by_segments(map,x,y,radius):
    candidate_segments = []
    close_by_segments = []
    distance = []
    already_perpendicular = []

    # Search map for close by nodes
    for node in map.railway_nodes:
        distance2 = (x - node.x)**2 + (y - node.y)**2

        if distance2 < radius**2:
            # axis.scatter(node.x,node.y,c='g')
            for way in map.nodes_to_segment_assignment[node.id]:
                if not way in candidate_segments:
                    candidate_segments.append(way)
    # print(candidate_segments)
    # for each segment belonging to close by nodes, check distance

    for way in candidate_segments:
        min_distance,min_point,orientation, arc_length_position = distance_to_segment(map,way,x,y)
        # print(arc_length_position)
        # print(map.railway_tracks[way].arc_length[-1])

        # if segment close by, keep it
        if min_distance < 60:
            # map.railway_tracks[way].plot_segment(axis,'r')
            # print(distance_to_segment(map,way,x,y))
            close_by_segments.append(way)
            distance.append(min_distance)
            # If projected point actually within track and not just at end, say True
            epsilon = 0
            if (0 - epsilon) <= arc_length_position < (map.railway_tracks[way].arc_length[-1] + epsilon):
                already_perpendicular.append(True)
            else:
                already_perpendicular.append(False)
    return close_by_segments, distance, already_perpendicular

def distance_to_segment(map,segment_id,x,y):
    min_distance = 100000
    min_point = np.array([ 0.0, 0.0 ])
    orientation_rad = 0
    p = np.array([ x, y ])
    arc_length_position = 0

    # For one track, check each linear piece,
    for i in range(1,len(map.railway_tracks[segment_id].x_cors)):
        # defining points for linear piece
        a = np.array([ map.railway_tracks[segment_id].x_cors[i-1], map.railway_tracks[segment_id].y_cors[i-1]])
        b = np.array([ map.railway_tracks[segment_id].x_cors[i], map.railway_tracks[segment_id].y_cors[i]])

        # point projected on linear piece
        res, frac = point_on_line(a, b, p)
        # distance
        distance = np.linalg.norm(p - res)
        # get the closest one
        if distance < min_distance:
            min_distance = distance
            min_point = res
            dif = min_point - p
            # orientation = np.arctan2(dif[1],dif[0]) # point to track
            orientation_rad = np.arctan2(b[1]-a[1],b[0]-a[0])
            length_of_segment = map.railway_tracks[segment_id].arc_length[i] - map.railway_tracks[segment_id].arc_length[i-1]
            arc_length_position = map.railway_tracks[segment_id].arc_length[i-1] + frac * length_of_segment
    return min_distance,min_point,orientation_rad, arc_length_position


def check_track_curvature(map, segment_id, x, y, idx):
    min_distance = 100000
    min_point = np.array([0.0, 0.0])
    orientation_rad = 0
    p = np.array([x, y])
    arc_length_position = 0


    a = np.array([map.railway_tracks[segment_id].x_cors[idx - 1], map.railway_tracks[segment_id].y_cors[idx - 1]])
    b = np.array([map.railway_tracks[segment_id].x_cors[idx], map.railway_tracks[segment_id].y_cors[idx]])

    orientation_rad = np.arctan2(b[1] - a[1], b[0] - a[0])

    return orientation_rad


def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    # if you need the the closest point belonging to the segment
    t = max(0, min(1, t))
    result = a + t * ab
    return result, t

def compute_curvature(p1,p2,p3):
    length_1 = np.linalg.norm(p1-p2)
    length_2 = np.linalg.norm(p1-p3)
    length_3 = np.linalg.norm(p2-p3)
    product_length = (length_1*length_2*length_3)
    if product_length == 0:
        return 0.0
    else:
        return 4 * area_of_triangle(p1,p2,p3) / (length_1*length_2*length_3)



def area_of_triangle(p1,p2,p3):
    vector_1 = p3 - p1
    vector_2 = p2 - p1
    area = 0.5 * np.linalg.norm(np.cross(vector_1,vector_2))
    return area


