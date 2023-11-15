#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import cv2
import time
import numpy as np
from rosbag import Bag
from rospy import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


path_to_bags = '/Volumes/T7 Shield/Cam2GPS Data/'

camera_bag_name = 'outside_2019-06-05-14-54-00.bag'
camera_imu_topic = 'imu_came'
image_0_topic = '/image_0' #'/bas_usb_0/image_raw'
image_1_topic = '/image_1' #'/bas_usb_1/image_raw'

gps_bag_name = 'synced_gps_imu_ekf.bag'
gps_topic = '/gps'
gps_imu_topic = '/imu_gps'
ekf_gps_topic = '/ekf_gps'


class BagData:

    def __init__(self,
                 path_to_bags: str,
                 gps_bag_name: str,
                 camera_bag_name: str = None, # will not initialize camera bag if no camera_bag_name is given
                 gps_pose_topic: str = '/gps',
                 image_0_topic: str = '/bas_usb_0/image_raw',
                 image_1_topic: str = '/bas_usb_1/image_raw',
                 ):

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
        '''
        Initializes additional useful information as bag attributes.
        '''
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
        '''
        Finds message of selected topic that is closest to the given time.

        Input:
        - bag: ROS Bag
        - topic: string of ROS topic name
        - time: float
        - freq_multiplier: int (used as a search region for the messages around the given time)

        Return:
        - message: ROS message
        - found_time: float in seconds
        - time_difference: float in seconds
        '''

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
        '''
        Input:
        - time: float in seconds
        - output_type: string 'numpy array' or 'dictionary'

        Output:
        - pose_array or pose_dict (according to output_type)
        - found_time: float in seconds
        '''

        msg, found_time, time_diff = self.__find_message_at_time__(self.gps_bag, self.gps_pose_topic, time)

        gps_pose_type = self.gps_bag.types_by_topic[self.gps_pose_topic]

        if gps_pose_type == 'nav_msgs/Odometry':
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

            # TODO: get heading (= yaw euler angle) from quaternions
            # from transformation import Transformation
            # Transformation.convert_quaternions(quaternion, to_type='euler angles')

            # TODO: make compatible with Frame/Keyframe/GPS objects for data import
        
        elif gps_pose_type == 'gps_common/GPSFix':
            print('not yet implemented GPS output type')

        elif gps_pose_type == 'geometry_msgs/PoseWithCovarianceStamped':
            print('not yet implemented EKF output type')

        else:
            print(f'type {gps_pose_type} not implemented')

        if output_type == 'numpy array':
            return pose_array, found_time
        elif output_type == 'dictionary':
            return pose_dict, found_time
        
    
    def convert_image_msg_to_cv2(self, msg: Image) -> np.ndarray:

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
        '''
        Input:
        - time: float in seconds

        Output:
        - image_0: numpy array
        - image_1: numpy array
        - found_time_0: float in seconds
        - found_time_1: float in seconds
        '''

        msg_0, found_time_0, time_diff_0 = self.__find_message_at_time__(self.camera_bag, self.image_0_topic, time)
        msg_1, found_time_1, time_diff_1 = self.__find_message_at_time__(self.camera_bag, self.image_1_topic, time)

        image_0 = self.convert_image_msg_to_cv2(msg_0)
        image_1 = self.convert_image_msg_to_cv2(msg_1)

        return image_0, image_1, found_time_0, found_time_1
    

    def find_gps_pose_and_images_at_time(self, time: float, pose_output_type: str = 'numpy array'):

        pose, found_time_pose = self.find_gps_pose_at_time(time, pose_output_type)
        image_0, image_1, found_time_image_0, found_time_image_1 = self.find_images_at_time(time)

        return pose, image_0, image_1, found_time_pose, found_time_image_0, found_time_image_1
    

    def save_gps_pose_and_images_at_time(self, time: float, path_to_poses: str, path_to_images_0: str, path_to_images_1: str, name: str = None):

        assert path_to_images_0 != path_to_images_1, print('path_to_images_0 and path_to_images_1 must be different')
        
        pose, image_0, image_1, found_time_pose, found_time_image_0, found_time_image_1 = self.find_gps_pose_and_images_at_time(time, 'numpy array')

        if not name:
            name = str(time).replace('.', '_')

        with open(path_to_poses + name + '.yaml', 'w') as outfile:
            yaml.dump(pose, outfile)

        cv2.imwrite(path_to_images_0 + name + '.jpg', image_0)
        cv2.imwrite(path_to_images_1 + name + '.jpg', image_1)


# Times of annotated stereo images
time_000490 = 1559739264.6531134
time_000570 = 1559739270.133156
time_000950 = 1559739302.7131376

test_image_time = 1559739296.962946


bag_data = BagData(path_to_bags, gps_bag_name, camera_bag_name)

path_to_images_0 = path_to_bags + 'images_0/'
path_to_images_1 = path_to_bags + 'images_1/'
path_to_poses = path_to_bags + 'poses/'


for time, name in zip([time_000490, time_000570, time_000950], ['000490', '000570', '000950']):

    print(f'Searching time {name}: {time}')

    pose_dict, found_time = bag_data.find_gps_pose_at_time(time, 'dictionary')
    print(f'Found pose time: {found_time}')
    print(f'Pose: {pose_dict}')
    
    with open(path_to_poses + name + '.yaml', 'w') as outfile:
            yaml.dump(pose_dict, outfile)
    
    found_images = bag_data.find_images_at_time(time)
    print(f'Found image times: {found_images[2]}, {found_images[3]}')

    cv2.imwrite(path_to_images_0 + name + '.jpg', found_images[0])
    cv2.imwrite(path_to_images_1 + name + '.jpg', found_images[1])
    
    print('---')
