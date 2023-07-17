#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Image, Imu
from custom_msgs.msg import orientationEstimate, positionEstimate, velocityEstimate
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
import os
import yaml
from cv_bridge import CvBridge, CvBridgeError
import cv2
import utm
from scipy.spatial.transform import Rotation as R
from gps_common.msg import GPSFix

class ImageGpsSync:
    def __init__(self):
        # Initialize everything
        rospy.init_node('image_gps_sync', anonymous=True)

        # Subscribers
        self.image0_sub = message_filters.Subscriber('bas_usb_0/image_raw', Image)
        self.image1_sub = message_filters.Subscriber('bas_usb_1/image_raw', Image)
        # self.odometry_sub = message_filters.Subscriber('odometry', Odometry)
        self.gps_sub = message_filters.Subscriber('oxts/gps_fix', GPSFix)

        # Publishers
        self.image0_pub = rospy.Publisher('sync/image_0', Image, queue_size=10)
        self.image1_pub = rospy.Publisher('sync/image_1', Image, queue_size=10)
        # self.odometry_pub = rospy.Publisher('sync/odometry', Odometry, queue_size=10)
        self.gps_pub = rospy.Publisher('sync/gps', GPSFix, queue_size=10)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            fs=[self.image0_sub, self.image1_sub, self.gps_sub],
            queue_size=60000,
            slop=0.005)
        
        self.ts.registerCallback(self.callback)

        # Output
        self.out_path = "/home/eric/Downloads/map_projection"
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        self.image0_path = os.path.join(self.out_path, "images_0")
        if not os.path.exists(self.image0_path):
            os.mkdir(self.image0_path)

        self.image1_path = os.path.join(self.out_path, "images_1")
        if not os.path.exists(self.image1_path):
            os.mkdir(self.image1_path)

        self.gps_pose_path = os.path.join(self.out_path, "gps_poses")
        if not os.path.exists(self.gps_pose_path):
            os.mkdir(self.gps_pose_path)

        # self.odmetry_pose_path = os.path.join(self.out_path, "odometry_poses")
        # if not os.path.exists(self.odmetry_pose_path):
        #     os.mkdir(self.odmetry_pose_path)

        self.image_counter = 0
        self.bridge = CvBridge()


    def callback(self, image_0, image_1, gps):
        print("Received synchronously! ")
        frame_name = f"{self.image_counter:06}"
        image0_path = os.path.join(self.image0_path, f"{frame_name}.jpg")
        image1_path = os.path.join(self.image1_path, f"{frame_name}.jpg")
        gps_yaml_path = os.path.join(self.gps_pose_path, f"{frame_name}.yaml")
        # odometry_yaml_path = os.path.join(self.odmetry_pose_path, f"{frame_name}.yaml")

        # Republish
        self.image0_pub.publish(image_0)
        self.image1_pub.publish(image_1)
        # self.odometry_pub.publish(odometry)
        self.gps_pub.publish(gps)

        # Save image
        for image, image_path in zip([image_0, image_1], [image0_path, image1_path]):
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            cv2.imwrite(image_path, cv_image)

        self.image_counter += 1

        # Get odometry pose
        # odometry_pose = dict()
        # odometry_pose["p_x"] = float(odometry.pose.pose.position.x)
        # odometry_pose["p_y"] = float(odometry.pose.pose.position.y)
        # odometry_pose["p_z"] = float(odometry.pose.pose.position.z)
        # odometry_pose["q_x"] = float(odometry.pose.pose.orientation.x)
        # odometry_pose["q_y"] = float(odometry.pose.pose.orientation.y)
        # odometry_pose["q_z"] = float(odometry.pose.pose.orientation.z)
        # odometry_pose["q_w"] = float(odometry.pose.pose.orientation.w)
        # odometry_pose["timestamp"] = odometry.header.stamp.secs + odometry.header.stamp.nsecs * 1e-9

        # rospy.loginfo(f"Odometry pose: {odometry_pose}")

        # with open(odometry_yaml_path, 'w') as outfile:
        #     yaml.dump(odometry_pose, outfile)

        # Get GPS pose
        gps_pose = dict()
        # Position
        latitude = gps.latitude
        longitude = gps.longitude
        altitude = gps.altitude
        utm_east, utm_north, utm_zone, utm_letter = utm.from_latlon(float(latitude), float(longitude))
        print(f"UTM East: {utm_east}, North: {utm_north}, Zone: {utm_zone}, Letter: {utm_letter}")
        # ENU (East-North-Up) convention
        gps_pose["p_x"] = float(utm_east) # longitude (East) is x
        gps_pose["p_y"] = float(utm_north) # latitude (North) is y
        gps_pose["p_z"] = altitude # 0
        # Orientation
        roll = gps.roll
        pitch = gps.pitch
        heading = 90 - gps.track # ENU heading is 0 at East, 90 at North, etc., while track is 0 at North, 90 at East
        gps_pose["heading"] = float(heading)
        rotation = R.from_euler('xyz', [roll, pitch, heading], degrees=True)
        quaternion = rotation.as_quat()
        gps_pose["q_x"] = float(quaternion[0])
        gps_pose["q_y"] = float(quaternion[1])
        gps_pose["q_z"] = float(quaternion[2])
        gps_pose["q_w"] = float(quaternion[3])
        gps_pose["timestamp"] = gps.header.stamp.secs + gps.header.stamp.nsecs * 1e-9
        
        rospy.loginfo(f"GPS Pose: {gps_pose}")

        with open(gps_yaml_path, 'w') as outfile:
            yaml.dump(gps_pose, outfile)
        

if __name__ == '__main__':
    ImageGpsSync()
    rospy.spin()
