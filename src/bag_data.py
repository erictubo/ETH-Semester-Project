#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles conversion of ROS bag data into synchronized frames for use in the camera localization pipeline.

This module provides functionality to read images, GPS, and IMU data from ROS bags and export them 
in a synchronized format suitable for further processing and annotation. It handles time synchronization,
data extraction, and frame export for the camera localization pipeline.

Classes:
    BagData: Main class for reading and synchronizing ROS bag data.
"""

import os
import yaml
import cv2
import time
import numpy as np
from rosbag import Bag
from rospy import Time
from sensor_msgs.msg import Image


# Directories to read from and write to
path_to_bags   = '/media/psf/Data/bags/'
path_to_frames = '/media/psf/frames_new/'

# ROS bag with camera data
camera_bag_name = 'outside_2019-06-05-14-54-00.bag'
camera_imu_topic = 'imu_camera'
image_0_topic = '/bas_usb_0/image_raw'
image_1_topic = '/bas_usb_1/image_raw'

# ROS bag with gps data
gps_bag_name = 'synced_gps_imu_ekf.bag'
gps_gps_topic = '/gps'
gps_imu_topic = '/imu_gps'
ekf_gps_topic = '/ekf_gps'


class BagData:
    """
    Handles reading and synchronizing data from ROS bag files for use in the pipeline.
    
    This class provides methods to extract GPS poses, images, and synchronize them for frame export.
    It supports both GPS-only and GPS+camera bag configurations, with automatic time synchronization
    and data format conversion.
    
    Attributes:
        gps_bag: ROS Bag object for GPS data
        camera_bag: ROS Bag object for camera data (optional)
        gps_pose_topic: Topic name for GPS pose data
        image_0_topic: Topic name for first camera image
        image_1_topic: Topic name for second camera image
    """

    def __init__(self,
                 path_to_bags: str,
                 gps_bag_name: str,
                 camera_bag_name: str = None, # will not initialize camera bag if no camera_bag_name is given
                 gps_pose_topic: str = '/gps', # alternative: ekf_gps_topic
                 image_0_topic: str = '/bas_usb_0/image_raw',
                 image_1_topic: str = '/bas_usb_1/image_raw',
                 ):
        """
        Initialize BagData with paths and topic names for GPS and camera bags.
        
        Args:
            path_to_bags: Directory path containing ROS bag files
            gps_bag_name: Name of the GPS bag file
            camera_bag_name: Name of the camera bag file (optional)
            gps_pose_topic: Topic name for GPS pose data (default: '/gps')
            image_0_topic: Topic name for first camera image (default: '/bas_usb_0/image_raw')
            image_1_topic: Topic name for second camera image (default: '/bas_usb_1/image_raw')
            
        Raises:
            AssertionError: If specified topics are not found in the bags
        """
        print('Initializing gps bag ...')
        self.gps_bag = Bag(path_to_bags + gps_bag_name, 'r')
        self.__init_more_bag_info__(self.gps_bag)
        print('GPS bag info:')
        print('- topics:', self.gps_bag.topics)
        print('- message types:', self.gps_bag.types_by_topic)
        print('- message counts:', self.gps_bag.counts_by_topic)
        print('- message frequencies:', self.gps_bag.freqs_by_topic)
        print('---')

        self.gps_pose_topic = gps_pose_topic

        assert self.gps_pose_topic in self.gps_bag.topics, print(f'gps_pose_topic: {self.gps_pose_topic} not found in gps_bag_topics: {self.gps_bag.topics}')


        # PROBLEM: camera bag initialization takes very long!
        if camera_bag_name:
            print('Initializing camera bag ...')
            start = time.time()
            self.camera_bag = Bag(path_to_bags + camera_bag_name, 'r')
            end = time.time()
            init_time = end - start
            print(f'Completed initialization in {round(init_time)} [s]')

            self.__init_more_bag_info__(self.camera_bag)
            print('Camera bag info:')
            print('- topics:', self.camera_bag.topics)
            print('- message types:', self.camera_bag.types_by_topic)
            print('- message counts:', self.camera_bag.counts_by_topic)
            print('- message frequencies:', self.camera_bag.freqs_by_topic)

            # self.combined_start_time = max(self.camera_bag.get_start_time(), self.gps_bag.get_start_time())
            # self.combined_end_time = min(self.camera_bag.get_end_time(), self.gps_bag.get_end_time())

            # print(f'Combined time range of bags: {self.combined_start_time} - {self.combined_end_time}')

            self.image_0_topic = image_0_topic
            self.image_1_topic = image_1_topic
        
            assert self.image_0_topic in self.camera_bag.topics, print(f'image_0_topic: {self.image_0_topic} not found in camera_bag_topics: {self.camera_bag.topics}')
            assert self.image_1_topic in self.camera_bag.topics, print(f'image_1_topic: {self.image_1_topic} not found in camera_bag_topics: {self.camera_bag.topics}')


    def __init_more_bag_info__(self, bag: Bag):
        """
        Initializes additional useful information as bag attributes.
        
        Args:
            bag: ROS Bag object to initialize
            
        Note:
            This method adds the following attributes to the bag object:
            - start_time: Bag start time in seconds
            - end_time: Bag end time in seconds
            - topics: List of topic names
            - types_by_topic: Dictionary mapping topics to message types
            - counts_by_topic: Dictionary mapping topics to message counts
            - freqs_by_topic: Dictionary mapping topics to message frequencies
        """
        bag.start_time: float = bag.get_start_time()
        bag.end_time: float = bag.get_end_time()

        bag.topics_info = bag.get_type_and_topic_info().topics

        bag.topics: list[str] = []
        bag.types_by_topic: dict[str: str] = {}
        bag.counts_by_topic: dict[str: int] = {}
        bag.freqs_by_topic: dict[str: float] = {}

        for topic, topic_tuple in bag.topics_info.items():
            bag.topics.append(topic)
            bag.types_by_topic[topic] = topic_tuple.msg_type
            bag.counts_by_topic[topic] = topic_tuple.message_count
            bag.freqs_by_topic[topic] = topic_tuple.frequency


    def __find_message_at_time__(self, bag: Bag, topic: str, time: float, freq_multiplier: int = 4):
        """
        Finds the message of a selected topic closest to the given time.
        
        Args:
            bag: ROS Bag object to search in
            topic: Topic name to search for
            time: Target time in seconds
            freq_multiplier: Search window multiplier (default: 4)
            
        Returns:
            tuple: (message, found_time, time_difference) or (None, None, None) if no message found
            
        Note:
            The search window is calculated as freq_multiplier / (2 * topic_frequency) around the target time.
            If the target time is outside the bag's time range, the entire bag is searched.
        """

        # TODO: either set bag timestamps equal to message timestamps or find way to read through header stamps

        topic_freq = bag.freqs_by_topic[topic]

        # If the bag time is the same as the topic header time
        if bag.start_time <= time <= bag.end_time:
            start_stamp = Time.from_sec(time - freq_multiplier/2/topic_freq)
            end_stamp = Time.from_sec(time + freq_multiplier/2/topic_freq)
        else:
            print(f'Time, {time}, not within time range of bag: {bag.start_time} - {bag.end_time}')
            print('Searching full time range ... (this may take a while for large bags)')
            start_stamp = None
            end_stamp = None
            

        best_time = None
        best_msg = None
        best_diff = -1
        for topic, msg, bag_stamp in bag.read_messages(topic, start_stamp, end_stamp):
            msg_stamp = msg.header.stamp
            # if bag_stamp != msg_stamp:
            #     print(f'Stamp of bag header and topic header are not equal: {bag_stamp} vs {msg_stamp}')            
            diff = abs(msg_stamp.to_sec() - time)
            if (diff < best_diff) or (best_diff == -1):
                best_diff = diff
                best_stamp = msg_stamp
                best_msg = msg
            best_time = best_stamp.to_sec()

        if not best_msg:
            print(f'No message found for topic {topic} within time range {start_stamp.to_sec()} - {end_stamp.to_sec()}')
            return None, None, None

        assert best_diff < 1.2/topic_freq,\
            print(f'Difference between given time {time} [s] and found time {best_time} [s] larger than expected: {best_diff} [s]')

        return best_msg, best_time, best_diff


    def find_gps_pose_at_time(self, time: float, output_type: str = 'numpy array'):
        """
        Find the GPS pose closest to the given time.
        
        Args:
            time: Target time in seconds
            output_type: Output format - 'numpy array' (default) or 'dictionary'
            
        Returns:
            tuple: (pose_data, found_time) where pose_data is either:
                - numpy array of shape (7,) [x, y, z, qx, qy, qz, qw] if output_type='numpy array'
                - dictionary with keys ['p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w', 'time'] if output_type='dictionary'
                
        Raises:
            NotImplementedError: If GPS pose message type is not supported
        """

        msg, found_time, time_diff = self.__find_message_at_time__(self.gps_bag, self.gps_pose_topic, time)

        gps_pose_type = self.gps_bag.types_by_topic[self.gps_pose_topic]

        if gps_pose_type in ['nav_msgs/Odometry', 'geometry_msgs/PoseWithCovarianceStamped']:
            p_x = msg.pose.pose.position.x
            p_y = msg.pose.pose.position.y
            p_z = msg.pose.pose.position.z
            q_x = msg.pose.pose.orientation.x
            q_y = msg.pose.pose.orientation.y
            q_z = msg.pose.pose.orientation.z
            q_w = msg.pose.pose.orientation.w

            position = np.array([p_x, p_y, p_z])
            quaternions = np.array([q_x, q_y, q_z, q_w])

            pose_array = np.array([p_x, p_y, p_z, q_x, q_y, q_z, q_w])
            pose_dict: dict[str:float] = {'p_x':p_x, 'p_y':p_y, 'p_z':p_z, 'q_x':q_x, 'q_y':q_y, 'q_z':q_z, 'q_w':q_w, 'time':found_time}

            # IDEA: make compatible with Frame/Keyframe/GPS objects for data import
        
        else:
            raise NotImplementedError(f'gps_pose_type {gps_pose_type} not implemented')

        if output_type == 'numpy array':
            return pose_array, found_time
        
        elif output_type == 'dictionary':
            return pose_dict, found_time
        
    
    def convert_image_msg_to_cv2(self, msg: Image) -> np.ndarray:
        """
        Convert ROS Image message to OpenCV format.
        
        Args:
            msg: ROS Image message
            
        Returns:
            OpenCV image as numpy array in BGR format
            
        Raises:
            AssertionError: If image encoding is not 'bayer_rggb8'
            
        Note:
            Currently only supports 'bayer_rggb8' encoding with Bayer to BGR conversion.
        """

        # msg.encoding      : string    # Encoding of pixels -- channel meaning, ordering, size
        # msg.height        : uint32    # image height, that is, number of rows
        # msg.width         : uint32    # image width, that is, number of columns
        # msg.is_bigendian  : unit8     # is this data bigendian?
        # msg.step          : uint32    # Full row length in bytes
        # msg.data          : uint8[]   # actual matrix data, size is (step * rows)

        assert msg.encoding == 'bayer_rggb8', print(f'Image encoding {msg.encoding} not implemented')
        image_bayer = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_BayerBG2BGR)

        return image_bgr

        
    def find_images_at_time(self, time: float):
        """
        Find stereo camera images closest to the given time.
        
        Args:
            time: Target time in seconds
            
        Returns:
            tuple: (image_0, image_1, found_time_0, found_time_1) where:
                - image_0: First camera image as numpy array
                - image_1: Second camera image as numpy array  
                - found_time_0: Actual time of first image in seconds
                - found_time_1: Actual time of second image in seconds
        """

        msg_0, found_time_0, time_diff_0 = self.__find_message_at_time__(self.camera_bag, self.image_0_topic, time)
        msg_1, found_time_1, time_diff_1 = self.__find_message_at_time__(self.camera_bag, self.image_1_topic, time)

        image_0 = self.convert_image_msg_to_cv2(msg_0)
        image_1 = self.convert_image_msg_to_cv2(msg_1)

        return image_0, image_1, found_time_0, found_time_1
    

    def find_gps_pose_and_images_at_time(self, time: float, pose_output_type: str = 'numpy array'):
        """
        Find synchronized GPS pose and stereo images at the given time.
        
        Args:
            time: Target time in seconds
            pose_output_type: GPS pose output format - 'numpy array' (default) or 'dictionary'
            
        Returns:
            tuple: (pose, image_0, image_1, found_time_pose, found_time_image_0, found_time_image_1) where:
                - pose: GPS pose data in specified format
                - image_0: First camera image as numpy array
                - image_1: Second camera image as numpy array
                - found_time_pose: Actual time of GPS pose in seconds
                - found_time_image_0: Actual time of first image in seconds
                - found_time_image_1: Actual time of second image in seconds
        """

        pose, found_time_pose = self.find_gps_pose_at_time(time, pose_output_type)
        image_0, image_1, found_time_image_0, found_time_image_1 = self.find_images_at_time(time)

        return pose, image_0, image_1, found_time_pose, found_time_image_0, found_time_image_1
    

    def export_frame(self, path_to_frames: str, time: float, name: str = None, print_info: bool = True):
        """
        Export a single synchronized frame with GPS pose and stereo images.
        
        Args:
            path_to_frames: Directory path to export frames to
            time: Target time in seconds
            name: Frame name (optional, defaults to time string)
            print_info: Whether to print export information (default: True)
            
        Note:
            Creates the following directory structure if it doesn't exist:
            - path_to_frames/images_0/ (first camera images)
            - path_to_frames/images_1/ (second camera images)  
            - path_to_frames/poses/ (GPS pose data)
            
            Exports:
            - GPS pose as YAML file: poses/{name}.yaml
            - First camera image as JPG: images_0/{name}.jpg
            - Second camera image as JPG: images_1/{name}.jpg
        """

        path_to_images_0 = path_to_frames + 'images_0/'
        path_to_images_1 = path_to_frames + 'images_1/'
        path_to_poses = path_to_frames + 'poses/'

        for path in [path_to_frames, path_to_images_0, path_to_images_1, path_to_poses]:
            if not os.path.exists(path):
                os.makedirs(path)

        if not name:
            name = str(time).replace('.', '_')

        if print_info:
            print(f'Searching time {name}: {time}')
        
        pose, image_0, image_1, found_time_pose, found_time_image_0, found_time_image_1 = self.find_gps_pose_and_images_at_time(time, 'numpy array')

        if print_info:
            print(f'Found pose time: {found_time_pose}')
            print(f'Pose: {pose}')
            print(f'Found image times: {found_time_image_0}, {found_time_image_1}')

        with open(path_to_poses + name + '.yaml', 'w') as outfile:
            yaml.dump(pose, outfile)

        cv2.imwrite(path_to_images_0 + name + '.jpg', image_0)
        cv2.imwrite(path_to_images_1 + name + '.jpg', image_1)

        if print_info:
            print('---')


    def export_frames(self, path_to_frames: str, times: list[float], names: list[str] = None, print_info: bool = True):
        """
        Export multiple synchronized frames with GPS poses and stereo images.
        
        Args:
            path_to_frames: Directory path to export frames to
            times: List of target times in seconds
            names: List of frame names (optional, must match length of times)
            print_info: Whether to print export information (default: True)
            
        Raises:
            AssertionError: If names is provided but has different length than times
            
        Note:
            Calls export_frame() for each time in the times list.
        """

        if names:
            assert len(times) == len(names), print('times and names must have the same length')

        for i in range(len(times)):
            self.export_frame(path_to_frames, times[i], names[i], print_info)


# 1. Annotate images
# 2. Export frames at timestamps of annotations
# 3. Run optimization with exported frames

times = [1559739264.6531134, 1559739270.133156, 1559739302.7131376]
names = ['000490', '000570', '000950']

bag_data = BagData(path_to_bags, gps_bag_name, camera_bag_name, gps_pose_topic=ekf_gps_topic)

bag_data.export_frames(path_to_frames, times, names, print_info=True)