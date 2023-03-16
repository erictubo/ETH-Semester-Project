#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nicolina Spiegelhalter 
E-mail: spiegeln@ethz.ch

"""

import cv2
import math
import yaml
import numpy as np
from vanishing_point_detection import get_vanishing_point as VanPoint
from track_information import get_track_information as TrackDet
from Yposition_estimate import y_estimate as YPos
from XZposition_estimate import xz_estimate as XZPos
from XRot_estimate import rot_x_estimate as XRot
from YRot_estimate import rot_y_estimate as YRot
from ZRot_estimate import rot_z_estimate as ZRot
from transformation_matrices import get_hom_transformation_matrices as TransMat
from matplotlib import image as plt

#class for each image frame with its attributes
class ImageAttributes:
    def __init__(self, image_nr, camera_nr, K):
        self.image_nr = image_nr
        self.camera_nr = camera_nr
        self.K = K
        self.img: np.ndarray[float]
        self.path: str
        self.path_poses: str
        self.line_info: list
        self.vanishing_point: list
        self.track_width_pix: float
        self.u_trackmiddle: float
        self.height_camera: float
        self.elevation: float
        self.height_gps: float
        self.H_w_cam: np.ndarray[float]
        self.H_gps_w: np.ndarray[float]
        self.H_gps_cam: np.ndarray[float]
        
    def get_path(self):
        self.path = str('/home/nicolina/catkin_ws/src/semester_thesis/bagfiles/bas_usb' + str(self.camera_nr) + '/images/')
        self.path_poses = str('/home/nicolina/catkin_ws/src/semester_thesis/bagfiles/bas_usb' + str(self.camera_nr) + '/poses/')
    
    def get_image_paths(self):
        nr_string = str(self.image_nr).replace('.0', '')
        if len(nr_string) < 6: 
            zeros_add = 6 - len(nr_string)
            image_path = str(nr_string.zfill(zeros_add+len(nr_string)) + '.jpg') 
            pose_path = str(nr_string.zfill(zeros_add+len(nr_string)) + '.yaml') 
        total_image_path = str(self.path + image_path)
        total_pose_path = str(self.path_poses + pose_path)
        return total_image_path, total_pose_path, image_path

    def get_tracks(self, canny_img, image_process):
        foundTracks, self.line_info, img_with_rails, self.track_width_pix, self.u_trackmiddle = TrackDet(canny_img, image_process)    
        return foundTracks, img_with_rails
        
    def get_vanishing_point(self, image_process, canny_img):
        self.vanishing_point, img_vanish = VanPoint(image_process, self.line_info, self.K, canny_img)
        return img_vanish

    def get_xrotation(self, image_process, canny_img):
        ang_x = XRot(image_process, self.K, canny_img, self.vanishing_point)
        return ang_x

    def get_yrotation(self, image_process, canny_img):
        ang_y = YRot(image_process, self.K, canny_img, self.vanishing_point)
        return ang_y
    
    def get_zrotation(self, canny_img, image_process):
        ang_z, img_rotz = ZRot(canny_img, image_process, self.u_trackmiddle, self.track_width_pix, self.vanishing_point, self.line_info)
        return ang_z, img_rotz
    
    def get_transformation_matrices(self, pose_estimates):
        self.H_w_cam, self.H_gps_w, self.H_gps_cam = TransMat(pose_estimates, self.image_nr, self.camera_nr)
        
    
    def get_yposition(self, image_process, canny_img, x_angle):
        i = self.image_nr
        y_pos, self.height_camera, self.height_gps, self.elevation = YPos(image_process, canny_img, self.track_width_pix, self.K, x_angle, i, self.path_poses, self.H_gps_w)
        return y_pos
    
    def get_xz_position(self, image_process, curr_heading, pose_estimates, train_pose_data, heading_list, straight_track_list, z_position_list, x_position_list):
        self.H_w_cam, self.H_gps_w, self.H_gps_cam = TransMat(pose_estimates, self.image_nr, self.camera_nr)
        foundXZ, straight_track_list, z_position_list, x_position_list = XZPos(image_process, curr_heading, pose_estimates, train_pose_data, heading_list, straight_track_list, z_position_list, x_position_list, self.height_camera, self.vanishing_point,self.image_nr, self.path, self.path_poses, self.K, self.elevation, self.H_w_cam, self.H_gps_w)
        return foundXZ, straight_track_list, z_position_list, x_position_list
        

'''booleans for order of estimates'''
findyPos = False 
findxzPos = False     
findxRot = True    
findyRot = True    
findzRot =  True   
isKartenProj = False 


'''initialization '''
x_position_list= []
y_position_list = []
z_position_list = []
x_angle_list = []
y_angle_list = []
z_angle_list = []
counter_ypos = 0
counter_xzpos = 0
counter_xang = 0
counter_yang = 0
counter_zang = 0

pose_estimates = [0,0,0,0,0,0] # [x_pos, y_pos, z_pos, x_ang (roll), y_ang (pitch), z_ang (yaw)]
heading_list = []
straight_track_list = []

'''thresholds how often a parameter should be computed'''
threshold_x_rot = 30
threshold_y_rot = 30
threshold_z_rot = 5
threshold_y_pos = 30
threshold_xz_pos = 5


'''Choose 0 for right camera and 1 for left camera '''
    
K = np.array([[698.8258566937119*2, 0, 492.9705850660823*2], 
              [0, 698.6495855393557*2, 293.3927615928415*2], 
              [0, 0, 1]])
    
camera_nr = 0

if camera_nr == 0:
    start = 500
    end = 1200
    
if camera_nr == 1 :
    start = 515
    end = 1200
    
number_of_images = np.linspace(start, end, num = (end-start+1))

for i in number_of_images:
    print(i)
    current_image = ImageAttributes(i, camera_nr, K)
    current_image.get_path()
    total_image_path, total_pose_path, image_path = current_image.get_image_paths()
    current_image.img = plt.imread(total_image_path)
    
    #get train pose from GPS data
    with open(total_pose_path, 'r') as stream:
        train_pose_data = yaml.safe_load(stream)
        
    if math.isnan(train_pose_data["heading"]) is False: # in case standing still
        curr_heading = train_pose_data["heading"]
        heading_list.append(curr_heading)
        straight_track_list.append(True)
        
        
        '''image pre-processing '''
        image_process = np.copy(current_image.img)
        image_copy = np.copy(current_image.img)
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)                                
        blur = cv2.GaussianBlur(gray, (3, 3), 0)                                    
        canny_img = cv2.Canny(blur, 35, 250)


        '''find Tracks'''
        foundTracks, img_with_rails = current_image.get_tracks(canny_img, image_process)
             
        
        '''find vanishing point '''
        if foundTracks is True:
            img_vanish = current_image.get_vanishing_point(image_process, canny_img)
    
    
        if findxRot and (foundTracks is True):
            '''find rotation z-axis estimate of camera'''
            ang_x = current_image.get_xrotation(image_process, canny_img)
            if ang_x != None:
                counter_xang += 1
            if counter_xang == threshold_x_rot+1:
                findxRot = False
                findyPos = True
        else:
            ang_x = None
        
        
        if findyRot and (foundTracks is True):
            '''find y rotation estimate of camera'''
            ang_y = current_image.get_yrotation(image_process, canny_img)
            if ang_y != None:
                counter_yang += 1
            if counter_yang == threshold_y_rot+1:
                findyRot = False      
        else:
            ang_y = None
            
            
        if findyPos and (foundTracks is True):
            '''find y axis estimate of camera'''
            current_image.get_transformation_matrices(pose_estimates)
            y_pos = current_image.get_yposition(image_process, canny_img, x_angle)
            if y_pos != None:
                counter_ypos += 1
            
            
        if findzRot and (foundTracks is True):
            '''find rotation z-axis estimate of camera'''
            ang_z,img_rotz = current_image.get_zrotation(canny_img, image_process)
            if ang_z != None:
                counter_zang += 1
                
            if counter_zang == threshold_z_rot+1:
                findzRot = False
                
        
        '''Collect parameter estimates '''
        #parameters received from image pipeline   
        if counter_ypos > 0 and counter_ypos < threshold_y_pos and y_pos != None:
            y_position_list.append(y_pos)
        if counter_xang > 0 and counter_xang < threshold_x_rot and ang_x != None:
            x_angle_list.append(ang_x)
        if counter_yang > 0 and counter_yang < threshold_y_rot and ang_y != None:
            y_angle_list.append(ang_y)
        if counter_zang > 0 and counter_zang < threshold_z_rot and ang_z != None:
            z_angle_list.append(ang_z)
            
            
        '''Compute median of found parameters'''
        if counter_xang == threshold_x_rot:
            x_angle = np.median(x_angle_list)
        if counter_ypos == threshold_y_pos:
            y_position = np.median(y_position_list)
        if counter_yang == threshold_y_rot:
            y_angle = np.median(y_angle_list)
        if counter_zang == threshold_z_rot: 
            z_angle = np.median(z_angle_list)
        if counter_xang >= threshold_x_rot and counter_yang >= threshold_y_rot and counter_zang > threshold_z_rot and counter_ypos > threshold_y_pos and findxzPos is False:
            pose_estimates = [0, y_position, 0, x_angle, y_angle, z_angle]
            findxzPos = True 
            
            
        '''Find x, z axis estimate of camera'''
        #after all estimates have sufficiently often been computed, except x and z position
        if findxzPos and (foundTracks is True):
            foundXZ, straight_track_list, z_position_list, x_position_list = current_image.get_xz_position(image_process, curr_heading, pose_estimates, train_pose_data, heading_list, straight_track_list, z_position_list, x_position_list)
            if foundXZ is True:
                counter_xzpos += 1

        if counter_xzpos >= threshold_xz_pos:
            z_position = np.median(z_position_list)
            pose_estimates[2] = z_position
            x_position =  np.median(x_position_list)
            pose_estimates[0] = x_position
            _, _, H_gps_cam = TransMat(pose_estimates, i, camera_nr)
            print('Found the pose between the GPS and the camera')
            break


                    

