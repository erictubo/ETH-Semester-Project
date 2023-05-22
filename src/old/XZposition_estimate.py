#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal: return horizontal and longitudinal distance between camera and gps

Input:
    img:                 Current frame
    curr_heading:        Heading of train at current position
    pose_estimates:      List of pose estimates [x_pos, y_pos, z_pos, x_ang, y_ang, z_ang]
    train_pose_data:     Train pose data
    heading_list:        List with all previous headings of the trian
    straight_track_list: List with Booleans whether train was on a straight track for last frames
    z_position_list:     List with estimated z_positions
    x_position_list:     List with estimated x_positions
    height_camera:       Camera height
    vanishing_point:     1x2 array with vanishing point coordinates in pixels
    image_nr:            Number of current image
    path:                Path of current frame
    path_poses:          Path of pose information of train at current frame
    elevation:           Elevation of ground at current train position
    K:                   Camera intrinsics
    H_gps_w:             Homogeneous transformation from GPS frame to world frame
    H_w_cam:             Homogeneous transformation from world frame to camera frame

Output:
    foundXZ:             Boolean whether estimate were found or not
    straight_track_list: List with Booleans whether train was on a straight track for the last frames, appended value for current frame
    z_position_list:     List with estimated z_positions
    x_position_list:     List with estimated x_positions
"""


import os
import cv2
import math
import zipfile
import numpy as np
import pandas as pd
import urllib.request
from preprocessing import image_preprocessing as ImgPrep
from numpy.linalg import inv


def xz_estimate(img, curr_heading, pose_estimates, train_pose_data, heading_list, straight_track_list, z_position_list, x_position_list, height_camera, vanishing_point, image_nr, path, path_poses, K, elevation, H_w_cam, H_gps_w):
    isImgVisualization = True
    image_pole = img 
    
    path_elevationdata = '/Users/eric/Developer/Cam2GPS/elevation/'
    data = pd.read_csv('/Users/eric/Developer/Cam2GPS/full_track.csv') #location of file with pole locations in world coordinates

    z_pos = None
    x_pos = None
    depth = None
    foundXZ = False
    pole_camview_gps = []
    fov_x = 2*np.arctan(img.shape[1]/(2*K[0][0]))
    height_pole = 5.3
    pitch = -pose_estimates[3]
    beginning_heading = heading_list[0]  
    
    if abs(curr_heading - beginning_heading) < 0.7: 
        straight_track_list.append(True)
        for colorfilter_counter in np.linspace(0, 11, 12):
            valid_lines = []
            img_white = ImgPrep(img, int(colorfilter_counter))   
            
            '''Segment image into components '''
            analysis = cv2.connectedComponentsWithStats(img_white, 4, cv2.CV_32S)
            (totalLabels, label_ids, values, centroid) = analysis
            output = np.zeros(img_white.shape, dtype="uint8")
            foundArea = False
            pixels_components = {}
            labels_valid_area = []
            for i in range(1, totalLabels):
                area = values[i, cv2.CC_STAT_AREA] 
                pixels_components[i] = [] #add label_id to list
                
                if (area > 2500) and (area < 13000):
                    foundArea = True
                    labels_valid_area.append(i)
    
                    (X, Y) = centroid[i]
                    
                    component = np.zeros(img_white.shape, dtype="uint8")
                    componentMask = (label_ids == i).astype("uint8") * 255
             
                    # Apply the mask using the bitwise operator
                    component = cv2.bitwise_or(component,componentMask)
                    output = cv2.bitwise_or(output, componentMask)
                    
                    
            if foundArea:        
                blur = cv2.GaussianBlur(output, (13, 13), 0)
                canny_img_2 = cv2.Canny(blur, 100, 200)
                
                '''Find lines in canny transformation of color filtered image '''
                lines = cv2.HoughLinesP(canny_img_2, 1, np.pi/180, 100, None, minLineLength=100, maxLineGap=7) #200, 50, 5 #250 70
                if lines is not None:
                    for i in range(0, len(lines)):
                        u1,v1,u2,v2 = lines[i,0,:]
                        if (np.arctan2(abs(v1-v2), abs(u1-u2)) > (85*np.pi/180)) and (np.arctan2(abs(v1-v2), abs(u1-u2)) < (95*np.pi/180)) and (u1 > 550 and u1 < 930):
                            if v1 > v2:
                                v_bottom = v1
                                u_bottom = u1
                                v_top = v2
                                u_top = u2
                                
                            else:
                                v_bottom = v2
                                u_bottom = u2
                                v_top = v1
                                u_top = u1
                            valid_lines.append([u_bottom, v_bottom, u_top, v_top]) #found line of component
                
                
                '''Find component the line belongs'''
                if len(valid_lines) != 0:
                    [u_bottom, v_bottom, u_top, v_top] = valid_lines[0]
                    
                    #find closest component to pixel
                    min_distance = math.inf
                    closest_component = 0
                    for i in labels_valid_area:
                        # get the centroid of the current component
                        [cx, cy] = centroid[i]
                        distance = math.sqrt((u_top - cx)**2 + (v_top - cy)**2)
                        
                        if distance < min_distance:
                            # update the closest component to be the current component
                            closest_component = i
                            min_distance = distance
                    
                    
                    #store pixels belonging to one component   
                    for j in range(img.shape[1]):
                        for i in range(img.shape[0]):
                            # get the component label for the current pixel
                            component_label = label_ids[i][j]
                    
                            # add the current pixel to the list of pixels for its component
                            if component_label < (totalLabels-1):
                                pixels_components[component_label+1].append((i, j))
                    
                    
                    '''Find lowest and highest pixel of mast '''
                    # initialize the lowest and highest pixel for the current component to the first pixel in the list
                    lowest_pixel = pixels_components[closest_component+1][0]
                    highest_pixel = pixels_components[closest_component+1][0]
                    
                    for pixel in pixels_components[closest_component+1]:
                        # if the y-coordinate of the current pixel is higher than the y-coordinate of the highest pixel, updated
                        if pixel[0] < highest_pixel[0]:
                            # update the highest pixel to be the current pixel
                            highest_pixel = pixel
                            
                    for pixel in pixels_components[closest_component+1]:  
                        if pixel[0] > lowest_pixel[0] and (abs(pixel[1] - highest_pixel[1]) < 5) :
                            # update the lowest pixel to be the current pixel
                            lowest_pixel = pixel
                            
                    # make sure pixels belong to same pole, is the most front pole, pole isn't cut off in image and lowest pixel lower than vanishing point       
                    if (abs(lowest_pixel[1] - highest_pixel[1]) < 7) and abs(lowest_pixel[0] - highest_pixel[0]) > 300 and highest_pixel[0] > 10 and lowest_pixel[0] > vanishing_point[1]: 
                        height_pole_pixel = lowest_pixel[0] - highest_pixel[0]
                        
                        #find closest pole ahead of train
                        df = pd.DataFrame(data, columns=['x', 'y'])
                        smallest_diff = math.inf
                        for index, pole in df.iterrows():  
                            test_pole_gpsview_world = [df.x[index], df.y[index], float(elevation), 1]
                            test_pole_gps = np.matmul(H_gps_w, test_pole_gpsview_world)
                            if smallest_diff > test_pole_gps[0] and test_pole_gps[0] > 0:     #only consider poles that are in front of GPS      
                                smallest_diff = test_pole_gps[0]
                                smallest_index = index

                        x_pole_world = df.x[smallest_index]
                        y_pole_world = df.y[smallest_index]
                         
                        #get elevation of pole position
                        file_name = str('dgm_33' + str(x_pole_world)[0:3] + '-' + str(y_pole_world)[0:4])
                        local_file_path = str(path_elevationdata + file_name)
                        if not os.path.exists(local_file_path):
                            zip_url = str('https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/' + file_name + '.zip' )
                            local_path_zip = str(path_elevationdata + file_name + '.zip')
                            urllib.request.urlretrieve(zip_url, local_path_zip)
                            with zipfile.ZipFile(local_path_zip, 'r') as zip_ref:
                                zip_ref.extractall(local_file_path)
                        
                        file = open(str(local_file_path+ '/' + file_name +'.xyz'), 'r')
                        diff_position = math.inf
                        for line in file:
                            split_string = line.split(",")
                            x_curr = split_string[0]
                            y_curr = split_string[1]
                            curr_difference = np.sqrt((x_pole_world - float(x_curr))**2 + (y_pole_world - float(y_curr))**2)
                            if curr_difference < diff_position:
                                new_elevation = split_string[2]
                                diff_position = curr_difference
                        elevation_pole = float(new_elevation)     
                        file.close()
                        
                        #define pole by elevation
                        pole_worldview_world = [x_pole_world, y_pole_world, float(elevation_pole), 1]
                        pole_worldview_gps = np.matmul(H_gps_w, pole_worldview_world)
                        
                        #compute real world pole height from depth
                        gamma = fov_x/img.shape[1]
                        angle_object = gamma* (img.shape[1]/2 - lowest_pixel[1]) - pose_estimates[4]
                        depth = (float(elevation_pole) - H_w_cam[2][3])/(H_w_cam[2][0]*((lowest_pixel[1]-K[0][2])/K[0][0])+H_w_cam[2][1] *((lowest_pixel[0]-K[1][2])/K[1][1]) + H_w_cam[2][2])
                        
                        diag_from_depth = np.sqrt((depth)**2 - (height_camera**2))
                        horizontal_from_depth = diag_from_depth * np.cos(angle_object)
                        height_pole_from_depth = height_pole_pixel * (horizontal_from_depth  + height_camera * np.tan(pitch))/K[1][1]
                        
                        pole_camview_cam = np.matmul(inv(K), [lowest_pixel[1]*depth, lowest_pixel[0]*depth, depth])
                        pole_camview_world = np.matmul(H_w_cam, [pole_camview_cam[0], pole_camview_cam[1], pole_camview_cam[2], 1])
                        pole_camview_gps = np.matmul(H_gps_w, pole_camview_world)

                        distance_pole_gps = np.sqrt((x_pole_world - train_pose_data['p_x'])**2 + (y_pole_world - train_pose_data['p_y'])**2) 
                        height_criterion = abs(height_pole_from_depth - height_pole) #check wether computed pole height fits ground truth
                        if  height_criterion < 0.5 and distance_pole_gps > 2:                        
                            z_pos = pole_worldview_gps[0] - pole_camview_gps[0]
                            x_pos = pole_camview_gps[1] - pole_worldview_gps[1]
                            
                            if z_pos < 0 or abs(z_pos) > 18:
                                z_pos = None
                            else:
                                x_position_list.append(x_pos)
                                z_position_list.append(z_pos)
                                foundXZ = True
    
                    # visualise highest and lowest pixel of pole, connect as line
                    image_pole = cv2.line(image_pole, (lowest_pixel[1], lowest_pixel[0]), (highest_pixel[1], highest_pixel[0]), (225, 0, 0), 4)
                
    else:
        straight_track_list.append(False) 
    
    '''Check if out of curve'''
    if len(heading_list) > 20: 
        prev_heading = heading_list[-19]
        if abs(curr_heading - beginning_heading) - abs(prev_heading - beginning_heading) < 0.4 and straight_track_list[-1] == False:
            heading_list = [curr_heading]  
            
    return foundXZ, straight_track_list, z_position_list, x_position_list, image_pole